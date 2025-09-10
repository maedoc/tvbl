# tvbl: a jupyter-lite based tvb


## Build locally

```sh
 micromamba create -f build-environment.yml -n build-tvbl
micromamba activate build-tvbl
jupyter lite build --contents content
```

## run locally

I use uv, the list of requirements are in uv-requirements.txt plus a few, the environment.yml is still best to look at

## ideas

most of want we want to do is possible 

- simulation
- sbi
- plotting

maybe we need interaction with storage, via

- ebrains-drive
- https://jupyter-server.readthedocs.io/en/latest/developers/contents.html to use ebrains-drive for storage?

maybe need pyunicore access?
