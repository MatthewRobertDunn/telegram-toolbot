"""The sandbox toolset.

Split by what the tools work on rather than by schema/implementation, so
each tool's schema sits above the function that runs it:

``shell``   run commands
``files``   read, write, edit, patch
``search``  grep, glob
``media``   read_image, send_file

``common`` holds the path helpers and per-user read tracking they share.
"""
