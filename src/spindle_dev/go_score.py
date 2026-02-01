import pandas as pd
import gseapy as gp
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path


def enrich_modules_with_gseapy(modules,
                              gene_sets=("Reactome_2022", "KEGG_2021_Human", "GO_Biological_Process_2023", "MsigDB_Hallmark_2020"),
                              organism="Human",
                              cutoff=0.05):
    import gseapy as gp
    import pandas as pd

    out = []
    for mi, module_genes in enumerate(modules):
        if len(module_genes) < 3:
            continue

        all_res = []
        for lib in gene_sets:
            enr = gp.enrichr(
                gene_list=module_genes,
                gene_sets=[lib],        # IMPORTANT: one at a time
                organism=organism,
                outdir=None,
                cutoff=cutoff
            )
            df = enr.results.copy()
            if df is None or len(df) == 0:
                continue

            # force library label (don’t trust server/client stamping)
            df["Gene_set"] = lib
            all_res.append(df)

        if len(all_res) == 0:
            continue

        res = pd.concat(all_res, ignore_index=True)
        out.append((mi, module_genes, res))

    return out


# def enrich_modules_with_gseapy(modules,
#                               gene_sets=("Reactome_2022", "KEGG_2021_Human", "GO_Biological_Process_2023","MSigDB_Hallmark_2020"),
#                               organism="Human",
#                               cutoff=0.05):

#     out = []
#     for mi, module_genes in enumerate(modules):
#         if len(module_genes) < 3:
#             continue

#         all_res = []
#         for lib in gene_sets:
#             enr = gp.enrichr(
#                 gene_list=module_genes,
#                 gene_sets=[lib],        # IMPORTANT: one at a time
#                 organism=organism,
#                 outdir=None,
#                 cutoff=cutoff
#             )
#             df = enr.results.copy()
#             if df is None or len(df) == 0:
#                 continue

#             # force library label (don’t trust server/client stamping)
#             df["Gene_set"] = lib
#             all_res.append(df)

#         if len(all_res) == 0:
#             continue

#         res = pd.concat(all_res, ignore_index=True)
#         out.append((mi, module_genes, res))

#     return out

def enrich_modules_with_gseapy(modules,
                              gene_sets=("Reactome_2022", "KEGG_2021_Human", "GO_Biological_Process_2023", "MSigDB_Hallmark_2020"),
                              organism="Human",
                              cutoff=0.05):
    import gseapy as gp
    import pandas as pd

    out = []
    for mi, module_genes in enumerate(modules):
        if len(module_genes) < 3:
            print(f"  Module {mi}: skipped (only {len(module_genes)} genes)")
            continue

        all_res = []
        for lib in gene_sets:
            try:
                enr = gp.enrichr(
                    gene_list=module_genes,
                    gene_sets=[lib],        # IMPORTANT: one at a time
                    organism=organism,
                    outdir=None,
                    cutoff=cutoff
                )
                df = enr.results.copy()
                if df is None or len(df) == 0:
                    print(f"  Module {mi}/{lib}: no significant results")
                    continue

                # force library label (don't trust server/client stamping)
                df["Gene_set"] = lib
                all_res.append(df)
            except Exception as e:
                # Catch EnrichrValidationError and other exceptions
                print(f"  Module {mi}/{lib}: {type(e).__name__}: {str(e)}")
                continue

        if len(all_res) == 0:
            print(f"  Module {mi}: no valid results from any gene set")
            continue

        res = pd.concat(all_res, ignore_index=True)
        out.append((mi, module_genes, res))
        print(f"  Module {mi}: enrichment completed with {len(res)} terms")

    return out


_GO_RE = re.compile(r"\s*\(GO[: ]?\d+\)\s*$", re.IGNORECASE)

def _clean_term_label(term: str, wrap_width: int = 42) -> str:
    term = _GO_RE.sub("", str(term)).strip()
    return "\n".join(textwrap.wrap(term, width=wrap_width))

def plot_module_enrichment_libraries(
    module_id: int,
    module_genes: list[str],
    res,
    top_n: int = 8,
    ncols: int = 2,
    bar_color: str = "#59A14F",
    wrap_width: int = 42,
    tick_fontsize: int = 7,
    label_fontsize: int = 8,
    title_fontsize: int = 10,
    compact_height_per_term: float = 0.20,
    min_fig_h: float = 2.4,
    fig_w: float = 11.0,          # wider default
    # saving
    save_dir: str | Path | None = None,
    fname_prefix: str = "enrich",
    fmt: str = "png",
    dpi: int = 200,
    close: bool = False,
):
    if res is None or len(res) == 0:
        return None, None
    if "Gene_set" not in res.columns:
        raise ValueError("Expected column 'Gene_set' in enrichr results DataFrame.")

    libs = sorted(res["Gene_set"].dropna().unique().tolist())
    print(libs)
    if len(libs) == 0:
        return None, None

    ncols = min(2, max(1, int(ncols)))
    nrows = math.ceil(len(libs) / ncols)

    fig_h = max(min_fig_h, compact_height_per_term * top_n * nrows + 0.7)

    # IMPORTANT: explicitly disable axis sharing
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        dpi=150,
        sharex=False,
        sharey=False
    )
    axes = np.array(axes).reshape(-1)

    used = 0
    for ax_i, lib_name in enumerate(libs):
        ax = axes[ax_i]
        ax.clear()  # reset any inherited/previous state

        df = res[res["Gene_set"] == lib_name].copy()
        if df.empty:
            ax.remove()
            continue

        if "Adjusted P-value" not in df.columns:
            raise ValueError("Expected 'Adjusted P-value' in enrichr results DataFrame.")

        df["score"] = -np.log10(df["Adjusted P-value"].astype(float) + 1e-300)
        df = df.sort_values("Adjusted P-value", ascending=True)
        df = df.drop_duplicates(subset=["Term"], keep="first")
        df = df.head(top_n).sort_values("score", ascending=True)

        terms = [_clean_term_label(t, wrap_width=wrap_width) for t in df["Term"].tolist()]
        scores = df["score"].to_numpy()

        ax.barh(terms, scores, color=bar_color)
        ax.set_xlabel(r"$-\log_{10}(\mathrm{FDR})$", fontsize=label_fontsize)
        ax.set_title(lib_name, fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(axis="x", alpha=0.2)
        ax.margins(x=0.02)

        # IMPORTANT: force y-limits to the correct number of bars for this axis
        ax.set_ylim(-0.5, len(terms) - 0.5)

        used += 1

    # remove leftover axes
    for j in range(len(libs), len(axes)):
        axes[j].remove()

    if used == 0:
        plt.close(fig)
        return None, None

    fig.suptitle(f"Module {module_id} (n={len(module_genes)})", fontsize=title_fontsize + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        outpath = save_dir / f"{fname_prefix}_module{module_id}_n{len(module_genes)}.{fmt}"
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(fig)

    return fig, axes


_OVERLAP_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")

def _parse_overlap(series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse Enrichr-style Overlap strings like '2/18' into (k, n, frac).
    Returns float arrays; frac is k/n.
    """
    k = np.full(len(series), np.nan, dtype=float)
    n = np.full(len(series), np.nan, dtype=float)
    for i, x in enumerate(series.astype(str).tolist()):
        m = _OVERLAP_RE.match(x)
        if m:
            k[i] = float(m.group(1))
            n[i] = float(m.group(2))
    frac = k / (n + 1e-12)
    return k, n, frac

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_module_enrichment_libraries_2(
    module_id: int,
    module_genes: list[str],
    res,                              # enrichr results DataFrame
    top_n: int = 8,
    ncols: int = 2,                   # <=2 as you requested
    wrap_width: int = 42,
    tick_fontsize: int = 7,
    label_fontsize: int = 8,
    title_fontsize: int = 10,
    compact_height_per_term: float = 0.20,
    min_fig_h: float = 2.4,
    fig_w: float = 11.0,
    # --- plot kind ---
    plot_kind: str = "bar",           # "bar" | "dot"
    # --- bar settings ---
    bar_color: str = "#59A14F",
    # --- dot settings ---
    cmap: str = "viridis",
    show_colorbar: bool = True,       # per-axis colorbar
    size_col: str = "Overlap",        # Enrichr column like "2/18"
    size_mode: str = "frac",          # "frac" (k/n) | "k" (numerator)
    size_range: tuple[float, float] = (18.0, 380.0),  # scatter s range (pt^2)
    size_alpha: float = 0.55,         # <1 dampens big bubbles
    size_clip_quantile: float = 0.98, # cap outliers in size_col
    show_size_legend: bool = True,
    size_legend_title: str = "Overlap",
    size_legend_n: int = 4,
    size_legend_loc: str = "lower right",
    # --- layout ---
    use_constrained_layout: bool = True,
    left_margin: float = 0.36,
    wspace: float = 0.55,
    hspace: float = 0.55,
    # --- saving ---
    save_dir: str | Path | None = None,
    fname_prefix: str = "enrich",
    fmt: str = "png",
    dpi: int = 200,
    close: bool = False,
):
    """
    Plot Enrichr enrichment results for a module, faceted by Gene_set/library.

    res must contain:
      - "Gene_set" (library name)
      - "Term"
      - "Adjusted P-value"
    Optional for dotplots:
      - "Overlap" as Enrichr-style strings like "2/18"
    """
    if res is None or len(res) == 0:
        return None, None

    required = {"Gene_set", "Term", "Adjusted P-value"}
    missing = required.difference(res.columns)
    if missing:
        raise ValueError(f"Missing required columns in res: {sorted(missing)}")

    libs = sorted(res["Gene_set"].dropna().unique().tolist())
    if len(libs) == 0:
        return None, None

    ncols = min(2, max(1, int(ncols)))
    nrows = math.ceil(len(libs) / ncols)

    fig_h = max(min_fig_h, compact_height_per_term * top_n * nrows + 0.7)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        dpi=150,
        sharex=False,
        sharey=False,
        constrained_layout=use_constrained_layout,
    )
    axes = np.array(axes).reshape(-1)

    used = 0

    for ax_i, lib_name in enumerate(libs):
        ax = axes[ax_i]
        ax.clear()

        df = res[res["Gene_set"] == lib_name].copy()
        if df.empty:
            ax.remove()
            continue

        # score = -log10(FDR)
        df["score"] = -np.log10(df["Adjusted P-value"].astype(float) + 1e-300)

        # keep best unique term (smallest FDR), take top_n, then plot lowest-to-highest for nice y order
        df = df.sort_values("Adjusted P-value", ascending=True)
        df = df.drop_duplicates(subset=["Term"], keep="first")
        df = df.head(top_n).sort_values("score", ascending=True)

        terms = [_clean_term_label(t, wrap_width=wrap_width) for t in df["Term"].tolist()]
        scores = df["score"].to_numpy(dtype=float)

        if plot_kind == "bar":
            ax.barh(terms, scores, color=bar_color)
            ax.set_xlabel(r"$-\log_{10}(\mathrm{FDR})$", fontsize=label_fontsize)

        elif plot_kind == "dot":
            y = np.arange(len(terms))

            # ---- compute values for bubble sizing
            vals = None
            if size_col is not None and size_col in df.columns:
                if str(size_col).lower() == "overlap":
                    k, n, frac = _parse_overlap(df[size_col])
                    vals = k if size_mode == "k" else frac
                else:
                    vals = df[size_col].to_numpy(dtype=float)

            if vals is None:
                vals = np.ones(len(scores), dtype=float)

            finite = np.isfinite(vals)
            if finite.any():
                cap = np.nanquantile(vals[finite], size_clip_quantile)
                vals = np.minimum(vals, cap)

                vmin = np.nanmin(vals[finite])
                vmax = np.nanmax(vals[finite])
            else:
                vmin, vmax = 0.0, 1.0

            # normalize -> [0, 1]
            t = (vals - vmin) / ((vmax - vmin) + 1e-12)
            t = np.clip(t, 0.0, 1.0)

            # dampen and map -> [smin, smax]
            smin, smax = size_range
            s = smin + (smax - smin) * (t ** float(size_alpha))
            s = np.nan_to_num(s, nan=smin)

            sc = ax.scatter(
                scores,
                y,
                s=s,
                c=scores,
                cmap=cmap,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.9,
            )

            ax.set_yticks(y)
            ax.set_yticklabels(terms)
            ax.set_xlabel(r"$-\log_{10}(\mathrm{FDR})$", fontsize=label_fontsize)

            if show_colorbar:
                # Use inset_axes for better control
                cax = inset_axes(
                    ax,
                    width="5%",      # thickness
                    height="60%",    # length
                    loc="upper right",
                    bbox_to_anchor=(0.20, 0, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
                cbar = plt.colorbar(sc, cax=cax)
                cbar.set_label(r"$-\log_{10}(\mathrm{FDR})$", fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=tick_fontsize)

            # ---- size legend (reference bubbles)
            if show_size_legend and finite.any():
                lo = np.nanmin(vals[finite])
                hi = np.nanmax(vals[finite])

                if size_legend_n < 2:
                    legend_vals = np.array([hi], dtype=float)
                else:
                    legend_vals = np.linspace(lo, hi, int(size_legend_n))

                lt = (legend_vals - vmin) / ((vmax - vmin) + 1e-12)
                lt = np.clip(lt, 0.0, 1.0)
                ls = smin + (smax - smin) * (lt ** float(size_alpha))

                # labels
                legend_title = size_legend_title
                if str(size_col).lower() == "overlap" and size_mode != "k":
                    legend_labels = [f"{100*v:.1f}%" for v in legend_vals]
                    legend_title = f"{size_legend_title} (k/n)"
                else:
                    legend_labels = [f"{v:.2g}" for v in legend_vals]

                handles = [
                    plt.Line2D(
                        [], [],
                        marker="o",
                        linestyle="",
                        markersize=float(np.sqrt(si)),  # markersize ~ diameter-ish
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                    )
                    for si in ls
                ]

                # leg = ax.legend(
                #     handles,
                #     legend_labels,
                #     title=legend_title,
                #     loc=size_legend_loc,
                #     frameon=True,
                #     borderpad=0.3,
                #     labelspacing=0.3,
                #     handletextpad=0.6,
                #     fontsize=tick_fontsize,
                #     title_fontsize=tick_fontsize,
                # )
                leg = ax.legend(
                    handles,
                    legend_labels,
                    title='Overlap',
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.15),   # push outside to the right
                    frameon=False,
                    borderpad=0.3,
                    labelspacing=0.3,
                    handletextpad=0.6,
                    fontsize=tick_fontsize,
                    title_fontsize=tick_fontsize,
                )
                # ax.add_artist(leg)

        else:
            raise ValueError(f"Unknown plot_kind={plot_kind!r}. Use 'bar' or 'dot'.")

        ax.set_title(lib_name, fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(axis="x", alpha=0.2)
        xmin, xmax = np.nanmin(scores), np.nanmax(scores)
        xr = xmax - xmin
        if xr <= 0:
            xr = max(1.0, xmax)

        pad = 0.10 * xr   # 15% padding on both sides
        ax.set_xlim(xmin - pad, xmax + pad)
        # ax.margins(x=1.02)
        ax.set_ylim(-0.5, len(terms) - 0.5)

        used += 1

    # remove leftover axes
    for j in range(len(libs), len(axes)):
        axes[j].remove()

    if used == 0:
        plt.close(fig)
        return None, None

    fig.suptitle(f"Module {module_id} (n={len(module_genes)})", fontsize=title_fontsize + 1)

    # layout control
    if not use_constrained_layout:
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    # fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.25, hspace=0.25)
    # fig.subplots_adjust(left=left_margin, wspace=wspace, hspace=hspace, top=0.90)

    # saving
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        outpath = save_dir / f"{fname_prefix}_module{module_id}_n{len(module_genes)}.{fmt}"
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        if close:
            plt.close(fig)

    return fig, axes
