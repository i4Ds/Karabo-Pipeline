External data
=============

Karabo provides helpers for downloading public data sets and caching them
locally. A resource is downloaded only when it is not already present in the
cache. The cache location is managed by :class:`karabo.util.file_handler.FileHandler`.

Use one of the supplied survey download objects when it matches the data set
you need, then call :meth:`~karabo.data.external_data.SingleFileDownloadObject.get`
to obtain the local path. For example:

.. code-block:: python

   from karabo.data.external_data import GLEAMSurveyDownloadObject

   gleam_catalogue_path = GLEAMSurveyDownloadObject().get()

For a custom remote file, create a
:class:`~karabo.data.external_data.SingleFileDownloadObject` with its path and
base URL. :class:`~karabo.data.external_data.ContainerContents` can discover
and download every file matching a regular expression in a remote container.

API reference
-------------

.. automodule:: karabo.data.external_data
   :members:
   :show-inheritance:
