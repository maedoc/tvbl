package main

// hub wires the config, relay, pair/job stores and guard together.

type hub struct {
	cfg   config
	relay *relay
}

func newHub(cfg config) *hub {
	return &hub{cfg: cfg, relay: newRelay()}
}
