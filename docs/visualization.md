# ExplainaBoard Visualization Tools

## Online Interface

The [web interface](https://explainaboard.inspiredco.ai) provides nice visualization
tools, browsing capabilities, etc. and make analysis relatively easy.

## Offline Tools

If you've preferred to use the offline [CLI interface](cli_interface.md), there are
also some rudimentary visualization tools at your disposal.

**Histograms:** If you want to draw visualizations of the bucketed analysis results
offline, you can run the following command over one or more reports:
```shell
python -m explainaboard.analyzers.draw_hist --reports report1.json report2.json
```

The results will be written out into the `figures` directory.