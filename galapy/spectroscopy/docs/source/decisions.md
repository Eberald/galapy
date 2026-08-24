# Modeling Decisions — CloudIA

This document summarizes the modeling decisions made in the CloudIA model.
_Format: one row per decision — number, subject, substance._

| # | Decision | Substance |
|---|---|---|
| 1 | CLOUDY = MC reprocessor | The molecular cloud is CLOUDY; GalaPy's native `ism.mc` radiative channel is switched off. |
| 2 | Recipes C & D | Continua from `save continuum` (col2/3/4/9); dust from `save grain abundance`. |
| 3 | GalaPy freezing | \(R_{\rm MC}\) is a module-level parameter, not read from `ism.mc`. |
| 4 | \(\xi_{d,\rm MW} = 0.45\) | Dust-to-metal gauge, not a measurement: fixes the origin of the \(\xi_d\) axis. |
| 5 | Dynamic \(\eta(\tau)\) routing | Which SSPs are still inside the cloud is decided by \(\eta(\tau)\), not a hard age cut. |
| 6 | \(f_{\rm cov} = 0.5\) (default) | Geometric covering fraction (Theulé+24 definition); liberatable via opt-in. |
| 7 | SSP library: PARSEC22.NT | **NT**, not NTL — NTL already embeds nebular emission (double-counting risk). |
| 8 | CLOUDY C25.00 | Production version; `cloudy_version` checked against the `.out` banner at runtime. |
| 9 | Mode A redshift constraint | Emission lines constrain \(z\) only when measured \(\lambda\) is available (\(\chi^2_{\rm position}\) term). |
| 10 | \(M_{\rm dust}\) — no double counting | \(M_{\rm tot} = M_{\rm cloud} + M_{\rm DD}\), combined only at reporting stage, never fed back into the likelihood. |
| 11 | Liberation matrix | 12 liberatable parameters; hard mutual exclusions (one liberation at a time, evidence-compared). |
| 12 | Grid charter | 7D+2 (HII), 8D+2 (PDR). Frozen at launch — changing it means full regeneration. |
| 13 | Line set | Frozen master list + per-run active subset; labels validated against official `LineList` files. |
| 14 | Chemical gauge: relative (N, C) / absolute (He) | Nicholls+17 sets \(\mathrm{N/O}\)–\(\mathrm{C/O}\) *variation* with \(Z\), zero-point stays GASS10. Theulé+24 Eq. 2 in absolute gauge — its \(\mathrm{O/H}\to0\) intercept is the measured \(Y_p\), not a convention. Anchoring He to GASS10 would give \(Y_p = 0.229\) (\(-7\%\)); the resulting \(+0.05\) dex offset at \(Z_\odot\) is the solar-photosphere settling correction, not an error. |
| 15 | \(\lambda\) without air/vacuum tag | Wavelength type `WL_NATIVE`: values follow the binary's own print convention (vacuum, C25 internal). |
| 16 | Agent on local Ollama | No external API calls; model digest recorded in run provenance. |
| 17 | PDR without \(\mathrm{H_2}\) | `database H2` not activated. Reopening this decision requires full PDR grid regeneration. |
| 18 | Single \(\sigma_v\) | Physical parameter of the source (not per-dataset nuisance); identifiable only with Mode B (raw spectrum). |