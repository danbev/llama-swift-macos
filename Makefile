download_model:
	wget https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_L.gguf?download=true

build:
	swift build

run:
	swift run

.PHONY: clean
clean:
	swift package reset
	swift package update
