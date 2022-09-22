#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates latex table for alignment pattern locations

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

LOCATIONS = [
    [],
    [6, 18],
    [6, 22],
    [6, 26],
    [6, 30],
    [6, 34],
    [6, 22, 38],
    [6, 24, 42],
    [6, 26, 46],
    [6, 28, 50],
    [6, 30, 54],
    [6, 32, 58],
    [6, 34, 62],
    [6, 26, 46, 66],
    [6, 26, 48, 70],
    [6, 26, 50, 74],
    [6, 30, 54, 78],
    [6, 30, 56, 82],
    [6, 30, 58, 86],
    [6, 34, 62, 90],
    [6, 28, 50, 72, 94],
    [6, 26, 50, 74, 98],
    [6, 30, 54, 78, 102],
    [6, 28, 54, 80, 106],
    [6, 32, 58, 84, 110],
    [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118],
    [6, 26, 50, 74, 98, 122],
    [6, 30, 54, 78, 102, 126],
    [6, 26, 52, 78, 104, 130],
    [6, 30, 56, 82, 108, 134],
    [6, 34, 60, 86, 112, 138],
    [6, 30, 58, 86, 114, 142],
    [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150],
    [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158],
    [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166],
    [6, 30, 58, 86, 114, 142, 170]
]

start = r"""\def\arraystretch{1.2}
\begin{center}
  \begin{longtabu}{|[2pt]c|c|c|c|c|c|c|c|[2pt]}
    \caption{Alignment pattern locations}
    \label{tab:qr_alignment}\\
    \tabucline[2pt]{-}
    Version & \multicolumn{7}{c|[2pt]}{Central x and y coordinates} \\
    \tabucline[2pt]{-}
    \endfirsthead
    \multicolumn{8}{r}{\emph{Continued from last page}}\\
    \hline
    Version & \multicolumn{7}{c|[2pt]}{Central x and y coordinates} \\
    \endhead
    Version & \multicolumn{7}{c|[2pt]}{Central x and y coordinates} \\
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
    with open("alignment.tex", "w") as f_tex:
        f_tex.write(start)
        
        for i, row in enumerate(LOCATIONS):
            if i > 0:
                f_tex.write("    \\hline\n")
            
            f_tex.write(f"    {i+1:2}")
            
            for j in range(7):
                val = row[j] if j < len(row) else ""
                f_tex.write(f" & {val:3}")
            
            f_tex.write(" \\\\\n")
        
        f_tex.write(end)