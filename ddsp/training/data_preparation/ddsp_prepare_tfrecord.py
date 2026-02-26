# Copyright 2026 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Create a TFRecord dataset from audio files.

Usage:
====================
ddsp_prepare_tfrecord \
--input_audio_filepatterns=/path/to/wavs/*wav,/path/to/mp3s/*mp3 \
--output_tfrecord_path=/path/to/output.tfrecord \
--num_shards=10 \
--alsologtostderr

"""

import threading
import time
from absl import app
from absl import flags
from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "input_audio_filepatterns",
    [],
    "List of filepatterns to glob for input audio files.",
)
flags.DEFINE_string(
    "output_tfrecord_path",
    None,
    "The prefix path to the output TFRecord. Shard numbers will be added to "
    "actual path(s).",
)
flags.DEFINE_integer(
    "num_shards",
    None,
    "The number of shards to use for the TFRecord. If None, this number will "
    "be determined automatically.",
)
flags.DEFINE_integer("sample_rate", 16000, "The sample rate to use for the audio.")
flags.DEFINE_integer(
    "frame_rate",
    250,
    "The frame rate to use for f0 and loudness features. If set to 0, "
    "these features will not be computed.",
)
flags.DEFINE_float(
    "example_secs",
    4,
    "The length of each example in seconds. Input audio will be split to this "
    "length using a sliding window. If 0, each full piece of audio will be "
    "used as an example.",
)
flags.DEFINE_float(
    "hop_secs",
    1,
    "The hop size between example start points (in seconds), when splitting "
    "audio into constant-length examples.",
)
flags.DEFINE_float(
    "eval_split_fraction",
    0.0,
    "Fraction of the dataset to reserve for eval split. If set to 0, no eval "
    "split is created.",
)
flags.DEFINE_float(
    "chunk_secs",
    20.0,
    "Chunk size in seconds used to split the input audio files. These "
    "non-overlapping chunks are partitioned into train and eval sets if "
    "eval_split_fraction > 0. This is used to split large audio files into "
    "manageable chunks for better parallelization and to enable "
    "non-overlapping train/eval splits.",
)
flags.DEFINE_boolean(
    "center",
    False,
    "Add padding to audio such that frame timestamps are centered. Increases "
    "number of frames by one.",
)
flags.DEFINE_boolean("viterbi", True, "Use viterbi decoding of pitch.")
flags.DEFINE_list(
    "pipeline_options",
    "--runner=DirectRunner,--direct_running_mode=in_memory",
    "A comma-separated list of command line arguments to be used as options "
    "for the Beam Pipeline.",
)
flags.DEFINE_boolean(
    "progress_bar",
    True,
    "Show progress information during execution.",
)
flags.DEFINE_integer(
    "max_workers",
    None,
    "Maximum number of workers for parallel processing. Default is auto.",
)
flags.DEFINE_string(
    "crepe_model",
    "tiny",
    "CREPE model size for pitch detection: tiny, small, medium, large, full. "
    "Smaller models are faster but less accurate.",
)
flags.DEFINE_boolean(
    "mixed_precision",
    False,
    "Use mixed precision (float16) for GPU acceleration. "
    "Requires GPU with tensor cores.",
)
flags.DEFINE_integer(
    "cache_size",
    1000,
    "Number of audio files to cache in memory for faster processing.",
)
flags.DEFINE_string(
    "direct_running_mode",
    "in_memory",
    "DirectRunner execution mode: in_memory, multi_threading, multi_processing. Use in_memory for large datasets.",
)
flags.DEFINE_integer(
    "wait_until_finish_duration",
    0,
    "Maximum time in milliseconds to wait for pipeline to finish. 0 means wait forever.",
)


class ProgressTracker:
    """Thread-safe progress tracker using logging."""

    def __init__(self, total, desc="Processing"):
        self._total = total
        self._completed = 0
        self._lock = threading.Lock()
        self._desc = desc
        self._start_time = time.time()
        self._last_report = 0

    def update(self, n=1):
        """Update progress by n items."""
        with self._lock:
            self._completed += n
            elapsed = time.time() - self._start_time
            if self._completed >= self._total or elapsed - self._last_report >= 5.0:
                self._last_report = elapsed
                rate = self._completed / elapsed if elapsed > 0 else 0
                remaining = (self._total - self._completed) / rate if rate > 0 else 0
                print(
                    f"{self._desc}: {self._completed}/{self._total} "
                    f"({100.0 * self._completed / self._total:.1f}%) - "
                    f"{rate:.1f} files/s - "
                    f"ETA: {remaining:.0f}s"
                )

    def completed(self):
        """Return the number of completed items."""
        with self._lock:
            return self._completed

    def total(self):
        """Return the total number of items."""
        return self._total


def run():
    input_audio_paths = []
    for filepattern in FLAGS.input_audio_filepatterns:
        input_audio_paths.extend(tf.io.gfile.glob(filepattern))

    total_files = len(input_audio_paths)

    if total_files == 0:
        print("No audio files found matching the provided patterns.")
        return

    print(f"Found {total_files} audio files to process.")

    if FLAGS.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Enabled mixed precision (float16) for GPU acceleration.")
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")

    pipeline_opts = list(FLAGS.pipeline_options)

    if FLAGS.max_workers is not None:
        max_workers_set = any("--max_num_workers" in opt for opt in pipeline_opts)
        if not max_workers_set:
            pipeline_opts.append(f"--max_num_workers={FLAGS.max_workers}")

    mode_set = any("--direct_running_mode" in opt for opt in pipeline_opts)
    if not mode_set and FLAGS.direct_running_mode:
        pipeline_opts.append(f"--direct_running_mode={FLAGS.direct_running_mode}")

    if FLAGS.wait_until_finish_duration > 0:
        wait_set = any("--wait_until_finish_duration" in opt for opt in pipeline_opts)
        if not wait_set:
            pipeline_opts.append(
                f"--wait_until_finish_duration={FLAGS.wait_until_finish_duration}"
            )

    progress_tracker = None
    if FLAGS.progress_bar and total_files > 0:
        progress_tracker = ProgressTracker(total_files, desc="Processing audio files")

    def update_progress():
        if progress_tracker:
            progress_tracker.update(1)

    prepare_tfrecord(
        input_audio_paths,
        FLAGS.output_tfrecord_path,
        num_shards=FLAGS.num_shards,
        sample_rate=FLAGS.sample_rate,
        frame_rate=FLAGS.frame_rate,
        example_secs=FLAGS.example_secs,
        hop_secs=FLAGS.hop_secs,
        eval_split_fraction=FLAGS.eval_split_fraction,
        chunk_secs=FLAGS.chunk_secs,
        center=FLAGS.center,
        viterbi=FLAGS.viterbi,
        crepe_model=FLAGS.crepe_model,
        pipeline_options=pipeline_opts,
        progress_callback=update_progress,
        direct_running_mode=FLAGS.direct_running_mode,
    )

    if progress_tracker:
        elapsed = time.time() - progress_tracker._start_time
        print(
            f"Successfully processed {progress_tracker.completed()} audio files "
            f"in {elapsed:.1f} seconds."
        )


def main(unused_argv):
    """From command line."""
    run()


def console_entry_point():
    """From pip installed script."""
    app.run(main)


if __name__ == "__main__":
    console_entry_point()
