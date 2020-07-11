;; testing mnist lesson-1 example to recognize digits

(ns fastai.lesson-1-mnist-exercise
  (:require [libpython-clj.python :as py :refer [py..]]
            [libpython-clj.require :refer [require-python]]

            [libpython-clj.metadata :as pymeta]
            [clojure.java.io :as io]))


(py/initialize!)

;; See https://forums.fast.ai/t/fastai-throwing-a-runtime-error-when-using-custom-train-test-sets/70262/30
(py.. (py/import-module "warnings") (filterwarnings "ignore"))

(require-python 'sys
                '[builtins :as python]
                'torch
                'fastai
                '[fastai.vision :as vision]
                '[fastai.vision.learner :as learner]
                '[fastai.vision.learner.models :as models]
                '[fastai.train :as train]
                '[matplotlib]
                '[matplotlib.pyplot :as plt]
                '[PIL :refer [Image]]
                '[numpy :as np]
                '[re]
                '[pandas :as pd])

(matplotlib/use "tkagg")

;; mnist

(def path
  (vision/untar_data (py.. vision/URLs -MNIST_SAMPLE)))

(map str (py.. (py.. path (joinpath "train")) (ls)))

(def tfms
  (vision/get_transforms :do_flip false))

(def data
  (py.. vision/ImageDataBunch (from_folder path
                                           :ds_tfms tfms
                                           :size 26)))

(py.. data (show_batch :rows 3 :figsize [5 5]))
(plt/show)

(def learn
  (learner/cnn_learner data models/resnet18 :metrics vision/accuracy))

(py.. learn (fit 2))

(def ClassificationInterpretation
  (py/get-attr (py/import-module "fastai.train") "ClassificationInterpretation"))

(def interp
  (py.. ClassificationInterpretation (from_learner learn)))

(py.. interp (plot_top_losses 9 :figsize [15 11]))
(plt/show)

(py.. interp (plot_confusion_matrix :figsize [12 12] :dpi 60))
(plt/show)

(py.. interp (most_confused :min_val 2))
