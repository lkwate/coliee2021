"""coliee_te_dataset dataset."""

import tensorflow_datasets as tfds
from . import coliee_te_dataset


class ColieeTeDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for coliee_te_dataset dataset."""
  # TODO(coliee_te_dataset):
  DATASET_CLASS = coliee_te_dataset.ColieeTeDataset
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
