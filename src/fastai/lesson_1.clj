(ns fastai.lesson-1
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

(plt/plot (vec (map #(Math/sin %) (range 20))))
(plt/show)

(pymeta/doc vision/untar_data)

(def path
  (vision/untar_data (py.. vision/URLs -PETS)))

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
(plt/show)

(py.. data -classes)

(= (count (py.. data -classes))
   (py.. data -c)
   37)

;; https://forums.fast.ai/t/fixing-notebook-1-convlearner-not-found/28367
;; https://forums.fast.ai/t/lesson-1-official-resources-and-updates/27936/7
;; ConvLearner replaced by cnn_learner
(def learn
  (learner/cnn_learner data models/resnet34 :metrics vision/error_rate))

(def first-time?
  true)

(when first-time?
  (py.. learn (fit_one_cycle 4))
  (py.. learn (save "stage-1")))

(when-not first-time?
  (py.. learn (load "stage-1")))

(def ClassificationInterpretation
  (py/get-attr (py/import-module "fastai.train") "ClassificationInterpretation"))

(def interp
  (py.. ClassificationInterpretation (from_learner learn)))

(py.. interp (plot_top_losses 9 :figsize [15 11]))
(plt/show)

(pymeta/doc (py.. interp -plot_top_losses))

(py.. interp (plot_confusion_matrix :figsize [12 12] :dpi 60))
(plt/show)

(py.. interp (most_confused :min_val 2))

(py.. learn (unfreeze))

(py.. learn (fit_one_cycle 1))

(py.. learn (load "stage-1"))

(py.. learn (lr_find))

(py.. learn -recorder (plot))
(plt/show)

(py.. learn (unfreeze))
(py.. learn (fit_one_cycle 2 :max_lr (python/slice 1e-6 1e-4)))


;; resnet 50

(def data
  (doto (py.. vision/ImageDataBunch (from_name_re (.getAbsolutePath path_img)
                                                  fnames
                                                  pat
                                                  :ds_tfms (vision/get_transforms)
                                                  :size 299
                                                  :bs 4))
    (py.. (normalize vision/imagenet_stats))))

(def learn
  (learner/cnn_learner data models/resnet50 :metrics vision/error_rate))

(py.. learn (fit_one_cycle 5))

(py.. learn (save "stage-1-50"))

(py.. learn (unfreeze))
(py.. learn (fit_one_cycle 1 :max_lr (python/slice 1e-6 1e-4)))

(py.. learn (load "stage-1-50"))

(def interp
  (py.. ClassificationInterpretation (from_learner learn)))

(py.. learn (most_confused :min_val 2))

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

(def df
  (pd/read_csv (py.. path (joinpath "labels.csv"))))

(py.. df (head))

;; mnist labels
(def data
  (py.. vision/ImageDataBunch (from_csv path
                                        :ds_tfms tfms
                                        :size 28)))

(py.. data (show_batch :rows 3 :figsize [5 5]))
(plt/show)
(py.. data -classes)

(def fn_paths
  (map #(py.. path (joinpath %)) (py/get-attr df "name")))

(take 2 fn_paths)

(def pat
  (re/compile (str #"(\d)/\d+[.]png")))

(def data
  (py.. vision/ImageDataBunch (from_name_re path
                                            fn_paths
                                            pat
                                            :ds_tfms tfms
                                            :size 24)))
(py.. data -classes)

(def data
  (py.. vision/ImageDataBunch (from_name_func path
                                              fn_paths
                                              :ds_tfms tfms
                                              :size 24
                                              :label_func (fn [x]
                                                            (let [x (str x)]
                                                              (if (re-find #"3" x)
                                                                "3"
                                                                "7"))))))

(py.. data -classes)


(def labels
  (map (fn [x] (let [x (str x)]
                 (if (re-find #"3" x)
                   "3"
                   "7")))
       fn_paths))

(take 5 labels)

(def data
  (py.. vision/ImageDataBunch (from_lists path
                                          fn_paths
                                          :ds_tfms tfms
                                          :size 24
                                          :labels labels)))

(py.. data -classes)
