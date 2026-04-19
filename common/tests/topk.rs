use common::topk::select_topk_by_vector;
use common::DistanceMetric;

#[test]
fn topk_returns_empty_for_k_zero() {
    let query = [0.0_f32, 0.0];
    let input = vec![(1_u32, &[2.0_f32, 0.0][..]), (2, &[0.2_f32, 0.0][..]), (3, &[0.5_f32, 0.0][..])];
    let got = select_topk_by_vector(&query, input, 0, DistanceMetric::L2);
    assert!(got.is_empty());
}

#[test]
fn topk_returns_all_when_k_exceeds_len_sorted() {
    let query = [0.0_f32, 0.0];
    let input = vec![(1_u32, &[0.8_f32, 0.0][..]), (2, &[0.2_f32, 0.0][..]), (3, &[0.5_f32, 0.0][..])];
    let got = select_topk_by_vector(&query, input, 10, DistanceMetric::L2);
    assert_eq!(got, vec![(2, 0.2), (3, 0.5), (1, 0.8)]);
}

#[test]
fn topk_picks_smallest_distances() {
    let query = [0.0_f32, 0.0];
    let input = vec![
        (11_u32, &[3.0_f32, 0.0][..]),
        (12, &[1.2_f32, 0.0][..]),
        (13, &[2.1_f32, 0.0][..]),
        (14, &[0.4_f32, 0.0][..]),
    ];
    let got = select_topk_by_vector(&query, input, 2, DistanceMetric::L2);
    assert_eq!(got, vec![(14, 0.4), (12, 1.2)]);
}

#[test]
fn topk_breaks_ties_by_smaller_id() {
    let query = [0.0_f32, 0.0];
    let input = vec![
        (30_u32, &[1.0_f32, 0.0][..]),
        (10, &[-1.0_f32, 0.0][..]),
        (20, &[0.0_f32, 1.0][..]),
        (40, &[0.5_f32, 0.0][..]),
    ];
    let got = select_topk_by_vector(&query, input, 3, DistanceMetric::L2);
    assert_eq!(got, vec![(40, 0.5), (10, 1.0), (20, 1.0)]);
}

#[test]
fn topk_uses_metric_for_ranking() {
    let query = [1.0_f32, 0.0];
    let input = vec![
        (1_u32, &[10.0_f32, 0.0][..]),
        (2, &[0.0_f32, 0.2][..]),
    ];

    let l2 = select_topk_by_vector(&query, input.clone(), 1, DistanceMetric::L2);
    let cosine = select_topk_by_vector(&query, input, 1, DistanceMetric::Cosine);

    assert_eq!(l2[0].0, 2);
    assert_eq!(cosine[0].0, 1);
}
