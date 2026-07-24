//! Dump an ONNX model's input/output names, types, and shapes.
//!
//!   cargo run --example introspect -- <path-to-model.onnx>
//!
//! ORT_DYLIB_PATH is provided automatically by .cargo/config.toml.

use ort::session::Session;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: cargo run --example introspect -- <model.onnx>");

    let mut builder = Session::builder().expect("session builder");
    let session = builder.commit_from_file(&path).expect("load model");

    println!("== {path} ==");
    println!("-- inputs --");
    for i in session.inputs() {
        println!("  {}: {:?}", i.name(), i.dtype());
    }
    println!("-- outputs --");
    for o in session.outputs() {
        println!("  {}: {:?}", o.name(), o.dtype());
    }
}
