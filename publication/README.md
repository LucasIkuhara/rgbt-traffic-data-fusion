# How to use

Add all source Markdown files to publication/src. They will be
appended in alphabetical order. 

To compile code, run `just pub`. 

Headers are defined in src/header.yml. This file is added to the
beginning of the Markdown file to give pandoc context. The date
is automatically interpolated.

## Custom templates

To use custom templates, the template.latex and style.cls files 
must be in the publication directory. The template.latex must be
adjusted to use whatever variables you set in header.yml and the
default template variables, like `$body$`. After adjustments, run
`just pub my_template`.
