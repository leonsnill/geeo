.PHONY: test docs-html docs-pdf docs check

# 1- run tests using your local EE authentication
test:
	pytest tests/

# 2 build html docu
docs-html:
	pandoc docs/documentation.md -s --toc \
		--metadata title="GEEO Documentation" \
		-c https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css \
		-o docs/documentation.html

# 3 build html version
docs-pdf:
	pandoc docs/documentation.md --pdf-engine=xelatex \
		-V geometry:margin=2cm --toc \
		-o docs/documentation.pdf

# create both
docs: docs-html docs-pdf

# entire pipeline
check: test docs