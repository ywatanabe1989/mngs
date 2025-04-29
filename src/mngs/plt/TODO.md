<!-- ---
!-- Timestamp: 2025-04-29 09:33:50
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/TODO.md
!-- --- -->

- [ ] Error handling, especially when not-implemented methods are accessed; this wrapper is not always compatible with the matplotlib.
- [ ] Implement test code to make it solid
- [ ] Increase coverage of available plot types


I mean:
fig, axes = mng.plt.subplots(...) # like matplotlib.pyplot.plots(...)
Then fig and axes have original methods written by me.
However, mngs.plt.subplots is not compllete but some methods and properties are not implemented for all matplotlib.

Thus, when not available methods are tried to process, I would like to handle them appropriately, using fallback systems with warning not implemented.

<!-- EOF -->