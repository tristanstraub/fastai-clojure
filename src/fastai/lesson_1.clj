(ns fastai.lesson-1
  (:require [libpython-clj.python :as py :refer [py..]]
            [libpython-clj.require :refer [require-python]]

            [libpython-clj.metadata :as pymeta]
            [clojure.java.io :as io]))

(py/initialize!)

;; See https://forums.fast.ai/t/fastai-throwing-a-runtime-error-when-using-custom-train-test-sets/70262/30
(py.. (py/import-module "warnings") (filterwarnings "ignore"))

(require-python 'sys
                'torch
                'fastai
                '[fastai.vision :as vision]
                '[fastai.vision.learner :as learner]
                '[fastai.vision.learner.models :as models]
                '[matplotlib]
                '[matplotlib.pyplot :as plt]
                '[PIL :refer [Image]]
                '[numpy :as np]
                '[re])

(matplotlib/use "tkagg")

(plt/plot (vec (map #(Math/sin %) (range 20))))
(plt/show)

(pymeta/doc vision/untar_data)

(def path (vision/untar_data (py.. vision/URLs -PETS)))

(map str (py.. path (ls)))

(def path_anno (io/file (str path) "annotations"))
(def path_img (io/file (str path) "images"))

(def fnames (vision/get_image_files (.getAbsolutePath path_img)))

(take 5 fnames)

(py.. np/random (seed 2))
(def pat (re/compile (str #"([^/]+)_\d+.jpg")))

(def data (doto (py.. vision/ImageDataBunch (from_name_re (.getAbsolutePath path_img)
                                                          fnames
                                                          pat
                                                          :ds_tfms (vision/get_transforms)
                                                          :size 224))
            (py.. (normalize vision/imagenet_stats))))

(py.. data (show_batch :rows 3 :figsize [7 6]))
#_ (plt/show)

(py.. data -classes)

(= (count (py.. data -classes))
   (py.. data -c)
   37)

;; https://forums.fast.ai/t/fixing-notebook-1-convlearner-not-found/28367
;; https://forums.fast.ai/t/lesson-1-official-resources-and-updates/27936/7
;; ConvLearner replaced by cnn_learner
(def learn (learner/cnn_learner data models/resnet34 :metrics vision/error_rate))

(py.. learn (fit_one_cycle 4))

(py.. learn (save "stage-1"))
