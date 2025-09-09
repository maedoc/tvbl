# tvbl: a jupyter-lite based tvb


## Build locally

```sh
 micromamba create -f build-environment.yml -n build-tvbl
micromamba activate build-tvbl
jupyter lite build --contents content
```

## run locally

I use uv, the list of requirements are in uv-requirements.txt
