# The following rules are copied from https://github.com/timtroendle/possibility-for-electricity-autarky/blob/master/rules/data-preprocessing.smk

configfile: "config/default.yaml"

URL_POP = "http://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GPW4_GLOBE_R2015A/GHS_POP_GPW42015_GLOBE_R2015A_54009_250/V1-0/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0.zip"

rule raw_population_zipped:
    message: "Download population data."
    output:
        protected("data/automatic/raw-population-data.zip")
    shell:
        "curl -sLo {output} '{URL_POP}'"


rule raw_population:
    message: "Extract population data as zip."
    input: rules.raw_population_zipped.output
    output: temp("build/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0.tif")
    shadow: "minimal"
    shell:
        """
        unzip {input} -d ./build/
        mv build/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0/GHS_POP_GPW42015_GLOBE_R2015A_54009_250_v1_0.tif {output}
        """


rule population_in_europe:
    message: "Clip population data to bounds of study."
    input:
        population = rules.raw_population.output,
    output:
        "build/population-europe.tif"
    params:
        bounds="{x_min},{y_min},{x_max},{y_max}".format(**config["scope"]["bounds"])
    shell:
        """
        rio clip --geographic --bounds {params.bounds} --co compress=LZW {input.population} -o {output}
        """