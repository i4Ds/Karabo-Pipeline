karabo.sourcedetection
======================

Overview
---------
Karabo uses `PyBDSF <https://www.astron.nl/citt/pybdsf/>`_ for source
detection. The result classes provide a Karabo-native representation of
detected sources, source-image products, and support for combining detections
from multiple images. ``SourceDetectionEvaluation`` compares detections with a
known sky model.

For a single image, call
:meth:`~karabo.sourcedetection.result.PyBDSFSourceDetectionResult.detect_sources_in_image`.
For multiple images that form a mosaic, use
:meth:`~karabo.sourcedetection.result.PyBDSFSourceDetectionResultList.detect_sources_in_images`.
Both return objects implementing ``ISourceDetectionResult``.

Source-detection results
------------------------

``SourceDetectionResult`` is the generic result container. The PyBDSF result
types wrap PyBDSF output and expose source catalogues and derived images in the
same format. ``PyBDSFSourceDetectionResultList`` merges detections from image
tiles and removes overlapping sources.

.. autoclass:: karabo.sourcedetection.result.ISourceDetectionResult
   :members:
   :special-members: __init__

.. autoclass:: karabo.sourcedetection.result.SourceDetectionResult
   :members:
   :special-members: __init__

.. autoclass:: karabo.sourcedetection.result.PyBDSFSourceDetectionResult
   :members:
   :special-members: __init__

.. autoclass:: karabo.sourcedetection.result.PyBDSFSourceDetectionResultList
   :members:
   :special-members: __init__

Evaluation
----------

``SourceDetectionEvaluation`` assigns detections to ground-truth source
positions and reports true-positive, false-positive, and false-negative
metrics.

.. autoclass:: karabo.sourcedetection.evaluation.SourceDetectionEvaluation
   :members:
   :special-members: __init__
