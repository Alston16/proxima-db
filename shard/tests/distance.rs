use shard::distance::{cosine_distance, l2_distance};

#[test]
fn l2_identical_vectors_is_zero() {
    let v = [1.0_f32, 2.0, 3.0];
    assert_eq!(l2_distance(&v, &v), 0.0);
}

#[test]
fn l2_known_value_unit_vectors() {
    // distance between (1,0) and (0,1) is sqrt(2)
    let a = [1.0_f32, 0.0];
    let b = [0.0_f32, 1.0];
    let got = l2_distance(&a, &b);
    assert!((got - 2.0_f32.sqrt()).abs() < 1e-6, "got {got}");
}

#[test]
fn l2_axis_aligned_distance() {
    // distance between (0,0,0) and (3,4,0) is 5
    let origin = [0.0_f32, 0.0, 0.0];
    let point = [3.0_f32, 4.0, 0.0];
    let got = l2_distance(&origin, &point);
    assert!((got - 5.0).abs() < 1e-6, "got {got}");
}

#[test]
fn cosine_identical_direction_is_zero() {
    let a = [2.0_f32, 0.0, 0.0];
    let b = [5.0_f32, 0.0, 0.0];
    let got = cosine_distance(&a, &b);
    assert!(got.abs() < 1e-6, "got {got}");
}

#[test]
fn cosine_opposite_direction_is_two() {
    let a = [1.0_f32, 0.0];
    let b = [-1.0_f32, 0.0];
    let got = cosine_distance(&a, &b);
    assert!((got - 2.0).abs() < 1e-6, "got {got}");
}

#[test]
fn cosine_orthogonal_vectors_is_one() {
    let a = [1.0_f32, 0.0];
    let b = [0.0_f32, 1.0];
    let got = cosine_distance(&a, &b);
    assert!((got - 1.0).abs() < 1e-6, "got {got}");
}

#[test]
fn cosine_zero_norm_vector_returns_one() {
    let zero = [0.0_f32, 0.0, 0.0];
    let other = [1.0_f32, 2.0, 3.0];
    assert_eq!(cosine_distance(&zero, &other), 1.0);
    assert_eq!(cosine_distance(&other, &zero), 1.0);
}
