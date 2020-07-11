(ns fastai.lesson-1
  (:require [libpython-clj.python :as py :refer [py..]]
            [libpython-clj.require :refer [require-python]]

            [libpython-clj.metadata :as pymeta]
            [clojure.java.io :as io]))

(py/initialize!)

(require-python 'sys
                'torch
                'fastai
                '[fastai.vision :as vision]
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
