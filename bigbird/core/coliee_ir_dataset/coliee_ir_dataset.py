"""coliee_ir_dataset dataset."""
"""
pip install git+https://github.com/google-research/bigbird.git -q
"""
from bigbird.core import flags
import tensorflow.compat.v2 as tf
import tensorflow_text as tft
import tensorflow_datasets as tfds
import sys

FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"): flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)

tf.enable_v2_behavior()

flags.DEFINE_string("data_dir", "data_dir", "")
flags.DEFINE_integer("max_encoder_length", 4096, "max encoder length")
flags.DEFINE_float("learning_rate", 1e-5, "learning rate")
flags.DEFINE_integer("num_train_steps", 10000, "num train steps")
flags.DEFINE_string("cls_token", "[SEP]", "")

FLAGS.data_dir = "tfds://imdb_reviews/plain_text"
FLAGS.attention_type = "block_sparse"
FLAGS.max_encoder_length = 4096  # 4096 on 16GB GPUs like V100, on free colab only lower memory GPU like T4 is available
FLAGS.learning_rate = 1e-5
FLAGS.vocab_model_file = "gpt2"
FLAGS.cls_token = "[SEP]"

bert_config = flags.as_dictionary()

tokenizer = tft.SentencepieceTokenizer(model=tf.io.gfile.GFile(FLAGS.vocab_model_file, "rb").read())


# TODO(coliee_ir_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(coliee_ir_dataset): BibTeX citation
_CITATION = """
"""


class ColieeIrDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for coliee_ir_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  filepath = os.path.abspath(os.path.dirname(__file__))
  pair_data = pd.read_csv(filepath + '/train_pair.csv')
  pair_data.sample(frac=1)
  train_data = pd.read_csv(filepath + '/train_data.csv')
  train_data = train_data.set_index('key')

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(coliee_ir_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'case': tfds.features.Tensor(shape=(1, FLAGS.max_encoder_length), dtype=tf.int32),
            'noticed_case': tfds.features.Tensor(shape=(1, FLAGS.max_encoder_length), dtype=tf.int32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(coliee_ir_dataset): Downloads the data and defines the splits

    # TODO(coliee_ir_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(),
    }

  def _generate_examples(self,):
    """Yields examples."""
    # TODO(coliee_ir_dataset): Yields (key, example) tuples from the dataset
    for row in self.pair_data.iterrows():
      id_case, id_noticed_case = row[1]['base_case'], row[1]['related_case']
      input_case, input_noticed_case = FLAGS.cls_token + self.train_data.loc[id_case]['content'], FLAGS.cls_token + self.train_data.loc[id_noticed_case]['content']
      # tokenization
      input_case = tokenizer.tokenize([input_case])
      input_case = input_case.to_tensor()
      pad_length = FLAGS.max_encoder_length - input_case.shape[-1]

      if pad_length > 0:
        pad = tf.constant([[0] * pad_length])
        input_case = tf.concat([input_case, pad], axis=-1)
      input_noticed_case = tokenizer.tokenize([input_noticed_case])
      pad_length = FLAGS.max_encoder_length - input_noticed_case.shape[-1]

      if pad_length > 0:
        pad = tf.constant([[0] * pad_length])
        input_noticed_case = tf.concat([input_noticed_case, pad], axis=-1)
      input_noticed_case = input_noticed_case.to_tensor()

      yield 'key', {
          'case': input_case.numpy(),
          'noticed_case': input_noticed_case.numpy(),
      }
