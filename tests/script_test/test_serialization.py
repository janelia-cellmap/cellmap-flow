import pytest
from cellmap_flow.utils.serilization_utils import get_process_dataset, serialize_norms_posts_to_json
from cellmap_flow.norm.input_normalize import MinMaxNormalizer, LambdaNormalizer
from cellmap_flow.post.postprocessors import ThresholdPostprocessor
import numpy as np

@pytest.fixture
def sample_data():
    return np.array([0.2, 0.4, 0.6, 0.8])

def test_serialization_and_deserialization_pipeline(sample_data):
    # Original pipeline components
    norms = [MinMaxNormalizer(), LambdaNormalizer("x*2-1")]
    posts = [ThresholdPostprocessor(threshold=0.5)]

    # Serialize to JSON
    json_data = serialize_norms_posts_to_json(norms, posts)
    assert isinstance(json_data, str)
    assert "MinMaxNormalizer" in json_data
    assert "LambdaNormalizer" in json_data
    assert "ThresholdPostprocessor" in json_data

    # Deserialize
    input_norm_fns, postprocess_fns = get_process_dataset(json_data)

    # Run data through original and deserialized pipeline
    # Original processing
    normed = sample_data.copy()
    for norm in norms:
        normed = norm(normed)
    processed_original = normed.copy()
    for post in posts:
        processed_original = post(processed_original)

    # Deserialized processing
    normed_deserialized = sample_data.copy()
    for norm_fn in input_norm_fns:
        normed_deserialized = norm_fn(normed_deserialized)
    processed_deserialized = normed_deserialized.copy()
    for post_fn in postprocess_fns:
        processed_deserialized = post_fn(processed_deserialized)

    # Assertions
    np.testing.assert_allclose(processed_original, processed_deserialized, rtol=1e-5)
