This is currently a paper scanner I'm working on to learn the OpenCV library. I'll add some demonstrations and more when it's done.

I've decided to try an epsilon neighborhood graph(ENG) approach to smoothing page detection, and the majority of the code files are now dedicated to supporting this feature. This repository still contains a working file scanner in pyscan.py, but the new purpose of this project is to study the ENG approach to page detection.

Without any testing, it's clear that this approach has dramatically improved the accuracy of the scanner.
