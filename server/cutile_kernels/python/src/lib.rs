use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::Arc;
use cutile_kernels::*;
use cutile::tensor::Tensor;
use cutile::api;
use cutile::DType;
use cuda_core::stream::Stream;
use cuda_core::device::Device;
use tokio::runtime::Runtime;

#[pyclass]
struct CutileContext {
    runtime: Runtime,
    device: Device,
    stream: Stream,
}

#[pymethods]
impl CutileContext {
    #[new]
    fn new(device_id: i32) -> PyResult<Self> {
        let runtime = Runtime::new().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let device = Device::new(device_id).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let stream = device.new_stream().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { runtime, device, stream })
    }

    fn masked_softmax(&self, py: Python, attention_scores: &PyAny, mask: &PyAny) -> PyResult<PyObject> {
        let attention_tensor = self.py_tensor_to_cutile(py, attention_scores)?;
        let mask_tensor = self.py_tensor_to_cutile(py, mask)?;
        
        let output = self.runtime.block_on(async {
            masked_softmax(&self.stream, &attention_tensor, &mask_tensor).await
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.cutile_tensor_to_py(py, &output)
    }

    fn q4_matmul(&self, py: Python, input: &PyAny, qweight: &PyAny, scales: &PyAny, qzeros: &PyAny, groupsize: i32, block_size_z: i32) -> PyResult<PyObject> {
        let input_tensor = self.py_tensor_to_cutile(py, input)?;
        let qweight_tensor = self.py_tensor_to_cutile(py, qweight)?;
        let scales_tensor = self.py_tensor_to_cutile(py, scales)?;
        let qzeros_tensor = self.py_tensor_to_cutile(py, qzeros)?;
        
        let output = self.runtime.block_on(async {
            q4_matmul(&self.stream, &input_tensor, &qweight_tensor, &scales_tensor, &qzeros_tensor, groupsize, block_size_z).await
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.cutile_tensor_to_py(py, &output)
    }

    fn q4_matmul_reconstruct(&self, py: Python, qweight: &PyAny, scales: &PyAny, qzeros: &PyAny, groupsize: i32) -> PyResult<PyObject> {
        let qweight_tensor = self.py_tensor_to_cutile(py, qweight)?;
        let scales_tensor = self.py_tensor_to_cutile(py, scales)?;
        let qzeros_tensor = self.py_tensor_to_cutile(py, qzeros)?;
        
        let output = self.runtime.block_on(async {
            q4_matmul_reconstruct(&self.stream, &qweight_tensor, &scales_tensor, &qzeros_tensor, groupsize).await
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.cutile_tensor_to_py(py, &output)
    }

    fn column_remap(&self, py: Python, input: &PyAny, remap: &PyAny) -> PyResult<PyObject> {
        let input_tensor = self.py_tensor_to_cutile(py, input)?;
        let remap_tensor = self.py_tensor_to_cutile(py, remap)?;
        
        let output = self.runtime.block_on(async {
            column_remap(&self.stream, &input_tensor, &remap_tensor).await
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        self.cutile_tensor_to_py(py, &output)
    }
}

impl CutileContext {
    fn py_tensor_to_cutile(&self, py: Python, obj: &PyAny) -> PyResult<Arc<Tensor<DType>>> {
        if let Ok(tensor) = obj.extract::<pyo3::Py<pyo3::types::PyModule>>() {
            if tensor.getattr(py, "__module__")?.extract::<String>()? == "torch" {
                return self.torch_tensor_to_cutile(py, obj);
            }
        }
        
        if let Ok(array) = obj.extract::<Vec<f32>>() {
            let shape = vec![array.len() as i32];
            let tensor = api::from_vec(array, &shape, DType::F32);
            return Ok(Arc::new(tensor));
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported tensor type"))
    }

    fn torch_tensor_to_cutile(&self, py: Python, obj: &PyAny) -> PyResult<Arc<Tensor<DType>>> {
        let data_ptr: usize = obj.call_method0(py, "data_ptr")?.extract(py)?;
        let shape: Vec<i64> = obj.call_method0(py, "shape")?.extract(py)?;
        let dtype_str: String = obj.call_method0(py, "dtype")?.extract(py)?;
        
        let dtype = match dtype_str.as_str() {
            "torch.float32" | "torch.float" => DType::F32,
            "torch.float16" | "torch.half" => DType::F16,
            "torch.bfloat16" => DType::BF16,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unsupported dtype: {}", dtype_str))),
        };
        
        let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        
        let tensor = unsafe {
            api::from_raw_ptr(data_ptr as *mut std::ffi::c_void, &shape_i32, dtype)
        };
        
        Ok(Arc::new(tensor))
    }

    fn cutile_tensor_to_py(&self, py: Python, tensor: &Arc<Tensor<DType>>) -> PyResult<PyObject> {
        let torch = py.import("torch")?;
        
        let shape: Vec<i64> = tensor.shape().iter().map(|&x| x as i64).collect();
        let dtype = match tensor.dtype() {
            DType::F32 => torch.getattr("float32")?,
            DType::F16 => torch.getattr("float16")?,
            DType::BF16 => torch.getattr("bfloat16")?,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Unsupported dtype")),
        };
        
        let data_ptr = tensor.as_ptr() as usize;
        
        let options = torch.call_method("TensorOptions", ())?.call_method("dtype", (dtype,))?.call_method("device", ("cuda",))?;
        let result = torch.call_method("from_blob", (data_ptr, shape, options))?.call_method("clone", ())?;
        
        Ok(result.into())
    }
}

#[pymodule]
fn cutile_kernels_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CutileContext>()?;
    Ok(())
}