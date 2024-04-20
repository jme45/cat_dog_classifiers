"""
Defines the classifiers for the dog/cat classification
"""
from pathlib import Path
from typing import Optional

from torchvision.transforms import v2 as transf_v2
from transfer_learning_vision_classifiers import classifiers as tlvs_classifiers


class CatDogClassifier(tlvs_classifiers.TransferLearningVisionClassifier):
    def __init__(
        self,
        model_type: str,
        class_names=["cat", "dog"],
        load_classifier_pretrained_weights: bool = False,
        classifier_pretrained_weights_file: Optional[str | Path] = None,
        data_augmentation_transforms: transf_v2.Transform = transf_v2.Identity(),
    ):
        """
        Initialise a binary classifier for cats and dogs.

        :param model_type: "vit_b_16", "vit_l_16", "effnet_b2", "effnet_b7", "trivial"
        :param class_names: Names for the two classes. Sensibly should be 'cat' and 'dog'
        :param load_classifier_pretrained_weights: whether to load classifier
        :param classifier_pretrained_weights_file: file for classifier data
        """
        self.class_names = class_names
        # Define train and prediction transform. Unlike for aircraft classification,
        # we don't need a transform to crop the authorship information
        self.train_transform = transf_v2.Compose(
            [
                tlvs_classifiers.TO_TENSOR_TRANSFORMS,
                data_augmentation_transforms,
                self.transforms,
            ]
        )
        self.predict_transform = transf_v2.Compose(
            [tlvs_classifiers.TO_TENSOR_TRANSFORMS, self.transforms]
        )
