use catgrad::core::nn::layers::*;
use catgrad::core::{Dtype, NdArrayType, Operation, Shape};
use open_hypergraphs::lax::{OpenHypergraph, functor::*, var, var::Var};

// Generate the residual arrow and save it
pub fn main() -> std::io::Result<()> {
    let arrow = residual_arrow();
    let arrow = var::forget::Forget.map_arrow(&arrow);
    crate::util::save_svg(&arrow, "images/residual.svg")
}

// A residual layer
pub fn residual(x: Var<NdArrayType, Operation>) -> Var<NdArrayType, Operation> {
    linear_layer("linear", x.clone()) + x
}

// A simplified linear layer for diagram purposes
fn linear_layer(name: &str, x: Var<NdArrayType, Operation>) -> Var<NdArrayType, Operation> {
    let state = x.state.clone();
    let n = *x.label.shape.0.last().unwrap();

    // A tensor type with...
    let ty = NdArrayType {
        shape: Shape(vec![n, n]), // N x N matrix
        dtype: Dtype::F32,        // Float32 entries
    };

    // Parameter value
    let p = parameter(&state, ty, name.to_string());
    mat_mul(&state, p, x.clone())
}

// Boilerplate to construct an OpenHypergraph using `residual`
fn residual_arrow() -> OpenHypergraph<NdArrayType, Operation> {
    let ty = NdArrayType::new(Shape(vec![8, 8]), Dtype::F32);
    var::build(|state| {
        let x = Var::new(state.clone(), ty.clone());
        let y = residual(x.clone());
        (vec![x], vec![y])
    })
    .unwrap()
}
