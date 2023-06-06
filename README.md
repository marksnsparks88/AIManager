# AIManager

An Logger for torch based machine learning projects. Currently tracks stats, numpy based arrays and model states.

For logging check out the bottom of AIManager.py for example usage

For plotting run AIManager.py to generate some example log data.

In a python prompt;

>>>plot = AIManager.Plots(root_dir='root_dir')

add a stat to the plot

>>>plot.add_stat('stat key')

add data to the plot. flattens a 2d array for each time step.

>>>plot.add_data('data_key')

display pyplot of added stats.optionally save image to plots folder

>>>plot.plot_stats(save=True/False)

add data to be displayed as an image.check help for parameters.

>>>add_image_data(**parameters)

render simgle image for testing.

>>>plots.render_image(ndx=0)

>>>plots.imfig.show()

reposition image data.check help for parameters

>>>plots.update_image_transforms(**parameters)

render simgle image for testing.

>>>plots.render_image(ndx=0)

>>>plots.imfig.show()

render final sequence to plots folder

>>>plot.render_image_sequence()
