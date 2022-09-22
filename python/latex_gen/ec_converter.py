#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates latex table for error correction information

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

levels = "LMQH"

start = r"""\def\arraystretch{1.5}
\begin{table}[H]
  \centering
  \begin{longtabu}{|[2pt]c|c|c|c|c|c|c|c|[2pt]}
    \tabucline[2pt]{-}
    Version & Correction level & Data codewords & Error correction codewords per block & Blocks in group 1 & Data codewords per group 1 blocks & Blocks in group 2 & Data codewords per group 2 blocks \\
    \tabucline[2pt]{-}
"""

end = r"""    \tabucline[2pt]{-}
  \end{longtabu}
  \caption{Error correction characteristics}
  \label{tab:qr_error_correction}
\end{table}
\def\arraystretch{1}
"""


start = r"""\def\arraystretch{1.2}
\begin{center}
  \begin{longtabu}{|[2pt]c|c|c|c|c|c|c|c|[2pt]}
    \caption{Error correction characteristics}
    \label{tab:qr_error_correction}\\
    \tabucline[2pt]{-}
    \rot{Version} & \rot{Correction level} & \rot{Data codewords} & \rot{\shortstack[l]{Error correction \\ codewords per block}} & \rot{Blocks in group 1} & \rot{\shortstack[l]{Data codewords per \\ group 1 blocks}} & \rot{Blocks in group 2} & \rot{\shortstack[l]{Data codewords per \\ group 2 blocks}} \\
    \tabucline[2pt]{-}
    \endfirsthead
    \multicolumn{8}{r}{\emph{Continued from last page}}\\
    \hline
    Ver & Level & Data CW & EC CW /B & Blocks G1 & CW G1 & Blocks G2 & CW G2 \\
    \endhead
    Ver & Level & Data CW & EC CW /B & Blocks G1 & CW G1 & Blocks G2 & CW G2 \\
    \hline
    \multicolumn{8}{r}{\emph{Continued on next page}}\\
    \endfoot
    \tabucline[2pt]{-}
    \endlastfoot
"""

end = r"""    \hline
  \end{longtabu}
\end{center}
\def\arraystretch{1}
"""

if __name__ == "__main__":
    with open("error_correction.txt", "r") as f_txt, open("error_correction.tex", "w") as f_tex:
        ecs = f_txt.read().split("\n\n")

        f_tex.write(start)

        """for i, version in enumerate(versions):
            lvls = version.split("\n")

            if i > 0:
                f_tex.write("    \\hline\n")

            f_tex.write(f"    \\multirow{{4}}{{*}}{{{i+1:2}}}")
            # f_tex.write(f"    {i+1:2}")

            for j, lvl in enumerate(lvls):
                values = " & ".join(
                    map(lambda s: f"{s:>4}", lvl.split("\t"))
                )

                if j > 0:
                    f_tex.write("                       ")
                    # f_tex.write("      ")

                f_tex.write(
                    f" & {levels[j]} & {values} \\\\{'*' if j < 3 else ''}\n")"""
        
        for i, ec in enumerate(ecs):
            lvls = [list(map(int, lvl.split("\t"))) for lvl in ec.split("\n")]
            lvls = [lvl + [0]*(6-len(lvl)) for lvl in lvls]
            
            if i > 0:
                f_tex.write("    \\hline\n")
            
            f_tex.write(f"    \\multirow{{4}}{{*}}{{{i+1:2}}}")
            
            for j, lvl in enumerate(lvls):
                values = " & ".join(
                    map(lambda s: f"{s:>4}", lvl)
                )
                
                if j > 0:
                    f_tex.write("                       ")
                    # f_tex.write("      ")

                f_tex.write(
                    f" & {levels[j]} & {values} \\\\{'*' if j < 3 else ''}\n")
        
        f_tex.write(end)
