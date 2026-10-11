# TF-71 oracle generator: runs bootsens ipw.R's get.extrema / extrema.os on PINNED fixture inputs
# and writes tests/causal/fixtures/ipw_oracle.json. The <expected> targets come ONLY from this R run
# (independence HARD RULE — never hand-filled or copied from the Python reimplementation, else the faithfulness gate is circular).
# Run via the pinned Docker image (see README); R is intentionally absent on dev hosts.

library(bootsens)  # GPL-3, installed in the pinned Docker image; NOT vendored into the repo

# --- pinned fixture (distinct Y -> unambiguous order(-Y); binary A with both classes) ---
A  <- c(1, 0, 1, 1, 0, 1, 0, 0)
X  <- matrix(c(0.5, -0.2, -1.0, 0.3, 0.8, 0.1, -0.3, 1.2,
               1.5, -0.7, -0.6, 0.4, 0.2, -1.1, 1.1, 0.9), ncol = 2, byrow = TRUE)
Y  <- c(0.90, 0.10, 0.70, 0.30, 0.55, 0.20, 0.80, 0.45)
fp <- c(0.60, 0.40, 0.55, 0.70, 0.30, 0.65, 0.35, 0.50)   # direct fitted.prob for the get.extrema cases

# (a)/(a') direct get.extrema
ge_all_g0 <- as.numeric(get.extrema(A, X, Y, gamma = 0, fitted.prob = fp, estimand = "all"))
ge_mis_g1 <- as.numeric(get.extrema(A, X, Y, gamma = 1, fitted.prob = fp, estimand = "missing"))

# (b)/(c) extrema.os(att). Emit the SAME propensity + outcome-regression the R path forms internally,
# so the Python injection seam is driven by identical inputs:
#   extrema.os(att) -> extrema.md(1-A, X, Y, "missing", reg.adjust): glm((1-A) ~ X); lm(Y ~ ., rows (1-A)==1).
flip     <- 1 - A
ps_flip  <- as.numeric(predict(glm(flip ~ X, family = "binomial"), type = "response"))
df       <- data.frame(Y = Y, X = X)
or_model <- lm(Y ~ ., df[flip == 1, ])
y_fitted <- as.numeric(predict(or_model, df))

os_noadj <- as.numeric(extrema.os(A, X, Y, gamma = 1, estimand = "att", reg.adjust = FALSE))
os_aipw  <- as.numeric(extrema.os(A, X, Y, gamma = 1, estimand = "att", reg.adjust = TRUE))

out <- list(
  provenance = list(
    source           = "bootsens @ github.com/qingyuanzhao/bootsens (GPL-3; installed in-image, not vendored)",
    generated_by     = "tools/oracle/ipw/gen_ipw_oracle.R (pinned rocker/r-ver:4.4.1 + remotes + jsonlite)",
    r_version        = R.version.string,
    bootsens_ref     = "6008a8d895fd8e5778bce24639d03fdfc492c66e",
    bootsens_version = as.character(utils::packageVersion("bootsens")),
    note             = "targets from an INDEPENDENT R run; never hand-filled or copied from the Python reimplementation (gate-2 non-circularity)"
  ),
  cases = list(
    list(name = "get_extrema_all_g0",     fn = "get.extrema",    estimand_all = TRUE,  gamma = 0,
         A = A, Y = Y, fitted_prob = fp,                           expected = ge_all_g0),
    list(name = "get_extrema_missing_g1", fn = "get.extrema",    estimand_all = FALSE, gamma = 1,
         A = A, Y = Y, fitted_prob = fp,                           expected = ge_mis_g1),
    list(name = "os_att_g1_noadj",        fn = "extrema.os.att", reg_adjust = FALSE,   gamma = 1,
         A = A, X = X, Y = Y, fitted_prob_flip = ps_flip,          expected = os_noadj),
    list(name = "os_att_g1_aipw",         fn = "extrema.os.att", reg_adjust = TRUE,    gamma = 1,
         A = A, X = X, Y = Y, fitted_prob_flip = ps_flip, y_fitted = y_fitted, expected = os_aipw)
  )
)

jsonlite::write_json(out, "/work/tests/causal/fixtures/ipw_oracle.json",
                     auto_unbox = TRUE, digits = 16, pretty = TRUE, matrix = "rowmajor")
cat("ipw_oracle.json written:", nrow(out$cases), "\n")
for (cs in out$cases) cat(sprintf("  %-22s expected = [%.10g, %.10g]\n", cs$name, cs$expected[1], cs$expected[2]))
