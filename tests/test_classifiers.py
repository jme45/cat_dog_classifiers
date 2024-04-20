"""test basic functionality"""

import PIL.Image
import numpy as np
import torch
import pytest

from cat_dog_classifiers import classifiers


def test_trivial_classifier(tmp_path):
    torch.manual_seed(0)

    # Instantiate a trivial classifier and save it
    trivial_classifier = classifiers.CatDogClassifier("trivial", ["cat", "dog"], False)
    save_path = tmp_path / "test.pth"
    trivial_classifier.save_model(save_path)

    # Now load this classifier again.
    trivial_classifier_loaded = classifiers.CatDogClassifier(
        "trivial", ["cat", "dog"], True, save_path
    )

    # Check that the linear layers are equal (the 2nd element in the Sequential).
    assert torch.equal(
        trivial_classifier.model.layers[1].weight,
        trivial_classifier_loaded.model.layers[1].weight,
    )


@pytest.fixture
def test_image():
    np.random.seed(0)
    img = PIL.Image.fromarray(np.uint8(255 * np.random.rand(300, 300, 3)))
    return img


def test_predict(test_image):
    torch.manual_seed(0)
    trivial_classifier = classifiers.CatDogClassifier("trivial", ["cat", "dog"], False)

    pred = trivial_classifier.predict(test_image, custom_predict_transform=None)
    expected_prod_cat = 0.40117138624191284
    assert np.isclose(pred[0]["cat"], expected_prod_cat)
