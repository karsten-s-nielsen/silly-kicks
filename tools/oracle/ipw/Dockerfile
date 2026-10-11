# TF-71 ipw.R oracle generator — pinned, dev-only (NOT in the wheel / CI-for-tests).
# Produces tests/causal/fixtures/ipw_oracle.json from the ZSB `bootsens` reference implementation,
# so the Python port's faithfulness gate (gate 2) compares against an INDEPENDENT R run, never the
# port itself. Repeatable like the lakehouse's Spark-local Docker + the das-reference pinned-oracle
# discipline.
#
# `bootsens` is GPL-3. It is INSTALLED into this ephemeral dev image at build time and is NEVER
# vendored into or distributed by the (MIT) silly-kicks repo — only the generated numeric fixture
# (facts, not copyrightable) is committed. This keeps the repo MIT-clean.
FROM rocker/r-ver:4.4.1

# remotes + jsonlite from a dated Posit CRAN snapshot (reproducible); bootsens pinned by commit SHA.
RUN R -e "options(repos='https://packagemanager.posit.co/cran/2024-08-01'); install.packages(c('remotes','jsonlite'))" \
 && R -e "options(repos='https://packagemanager.posit.co/cran/2024-08-01'); remotes::install_github('qingyuanzhao/bootsens', ref='6008a8d895fd8e5778bce24639d03fdfc492c66e', upgrade='never')" \
 && R -e "stopifnot(requireNamespace('bootsens', quietly=TRUE), requireNamespace('jsonlite', quietly=TRUE))"

WORKDIR /work
CMD ["Rscript", "/work/tools/oracle/ipw/gen_ipw_oracle.R"]
