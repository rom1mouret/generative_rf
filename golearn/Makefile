OUT_DIR=$(shell go env GOPATH)/src/github.com/rom1mouret/generative_rf/golearn/generative_rf

all: install
install:
	go get -v github.com/sjwhitworth/golearn/ensemble
	mkdir -p $(OUT_DIR)
	cp generative_rf/* $(OUT_DIR)/
	(cd $(OUT_DIR) && go install -i && echo "SUCCESS")

run: install
	(cd example && go run main.go trainset.csv)
