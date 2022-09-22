#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates latex table for version capacity information

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

levels = "LMQH"

start = r"""\def\arraystretch{1.5}
\begin{table}[H]
  \centering
  \begin{longtabu}{|[2pt]c|c|c:c:c:c|[2pt]}
    \tabucline[2pt]{-}
    Version & Correction level & Numerical & Alphanumerical & Byte & Kanji \\
    \tabucline[2pt]{-}
"""

end = r"""    \tabucline[2pt]{-}
  \end{longtabu}
  \caption{Version capacities}
  \label{tab:qr_versions}
\end{table}
\def\arraystretch{1}
"""


start = r"""\def\arraystretch{1.2}
\begin{center}
  \begin{longtabu}{|[2pt]c|c|c:c:c:c|[2pt]}
    \caption{Version capacities}
    \label{tab:qr_versions}\\
    \tabucline[2pt]{-}
    Version & Correction level & Numerical & Alphanumerical & Byte & Kanji \\
    \tabucline[2pt]{-}
    \endfirsthead
    \multicolumn{6}{r}{\emph{Continued from last page}}\\
    \endhead
    \multicolumn{6}{r}{\emph{Continued on next page}}\\
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
    with open("qr_versions.txt", "r") as f_txt, open("qr_versions.tex", "w") as f_tex:
        versions = f_txt.read().split("\n\n")
        
        f_tex.write(start)
        
        for i, version in enumerate(versions):
            lvls = version.split("\n")
            
            if i > 0:
                f_tex.write("    \\hline\n")
            
            f_tex.write(f"    \\multirow{{4}}{{*}}{{{i+1:2}}}")
            #f_tex.write(f"    {i+1:2}")
            
            for j, lvl in enumerate(lvls):
                values = " & ".join(
                    map(lambda s: f"{s:>4}", lvl.split("\t"))
                )
                
                if j > 0:
                    f_tex.write("                       ")
                    #f_tex.write("      ")
                
                f_tex.write(f" & {levels[j]} & {values} \\\\{'*' if j < 3 else ''}\n")
        
        f_tex.write(end)