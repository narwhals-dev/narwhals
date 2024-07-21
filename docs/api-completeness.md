# API Completeness

Narwhals has two different level of support for libraries: "full" and "interchange".

Libraries for which we have full support we intend to support the whole Narwhals API, however this is a work in progress.

In the following table it is possible to check which method is implemented for which backend. 

!!! info

    - "pandas-like" means pandas, cuDF and Modin
    - Polars supports all the methods (by design)

| Cla | Met | pan | arr |
| ss  | hod | das | ow  |
|     |     | -li |     |
|     |     | ke  |     |
|-----|-----|-----|-----|
| Dat | clo | :wh | :wh |
| aFr | ne  | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | col | :wh | :wh |
| aFr | lec | ite | ite |
| ame | t_s | _ch | _ch |
|     | che | eck | eck |
|     | ma  | _ma | _ma |
|     |     | rk: | rk: |
| Dat | col | :wh | :wh |
| aFr | umn | ite | ite |
| ame | s   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | dro | :wh | :wh |
| aFr | p   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | dro | :wh | :wh |
| aFr | p_n | ite | ite |
| ame | ull | _ch | _ch |
|     | s   | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | fil | :wh | :wh |
| aFr | ter | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | get | :wh | :wh |
| aFr | _co | ite | ite |
| ame | lum | _ch | _ch |
|     | n   | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | gro | :wh | :wh |
| aFr | up_ | ite | ite |
| ame | by  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | hea | :wh | :wh |
| aFr | d   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | is_ | :wh | :x: |
| aFr | dup | ite |     |
| ame | lic | _ch |     |
|     | ate | eck |     |
|     | d   | _ma |     |
|     |     | rk: |     |
| Dat | is_ | :wh | :wh |
| aFr | emp | ite | ite |
| ame | ty  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | is_ | :wh | :x: |
| aFr | uni | ite |     |
| ame | que | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Dat | ite | :wh | :wh |
| aFr | m   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | ite | :wh | :x: |
| aFr | r_r | ite |     |
| ame | ows | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Dat | joi | :wh | :wh |
| aFr | n   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | laz | :wh | :wh |
| aFr | y   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | nul | :wh | :wh |
| aFr | l_c | ite | ite |
| ame | oun | _ch | _ch |
|     | t   | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | pip | :x: | :x: |
| aFr | e   |     |     |
| ame |     |     |     |
| Dat | ren | :wh | :wh |
| aFr | ame | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | row | :wh | :wh |
| aFr | s   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | sch | :wh | :wh |
| aFr | ema | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | sel | :wh | :wh |
| aFr | ect | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | sha | :wh | :wh |
| aFr | pe  | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | sor | :wh | :wh |
| aFr | t   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | tai | :wh | :wh |
| aFr | l   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | to_ | :wh | :wh |
| aFr | dic | ite | ite |
| ame | t   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | to_ | :wh | :wh |
| aFr | num | ite | ite |
| ame | py  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | to_ | :wh | :wh |
| aFr | pan | ite | ite |
| ame | das | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | uni | :wh | :x: |
| aFr | que | ite |     |
| ame |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Dat | wit | :wh | :wh |
| aFr | h_c | ite | ite |
| ame | olu | _ch | _ch |
|     | mns | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Dat | wit | :wh | :wh |
| aFr | h_r | ite | ite |
| ame | ow_ | _ch | _ch |
|     | ind | eck | eck |
|     | ex  | _ma | _ma |
|     |     | rk: | rk: |
| Dat | wri | :wh | :wh |
| aFr | te_ | ite | ite |
| ame | par | _ch | _ch |
|     | que | eck | eck |
|     | t   | _ma | _ma |
|     |     | rk: | rk: |
| Exp | abs | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | ali | :wh | :wh |
| r   | as  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | all | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | any | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | cas | :wh | :wh |
| r   | t   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | cat | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | cou | :wh | :wh |
| r   | nt  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | cum | :wh | :wh |
| r   | _su | ite | ite |
|     | m   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | dif | :wh | :wh |
| r   | f   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | dro | :wh | :x: |
| r   | p_n | ite |     |
|     | ull | _ch |     |
|     | s   | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | dt  | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | fil | :wh | :wh |
| r   | l_n | ite | ite |
|     | ull | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | fil | :wh | :wh |
| r   | ter | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | hea | :wh | :wh |
| r   | d   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | is_ | :wh | :x: |
| r   | bet | ite |     |
|     | wee | _ch |     |
|     | n   | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | is_ | :wh | :x: |
| r   | dup | ite |     |
|     | lic | _ch |     |
|     | ate | eck |     |
|     | d   | _ma |     |
|     |     | rk: |     |
| Exp | is_ | :wh | :x: |
| r   | fir | ite |     |
|     | st_ | _ch |     |
|     | dis | eck |     |
|     | tin | _ma |     |
|     | ct  | rk: |     |
| Exp | is_ | :wh | :wh |
| r   | in  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | is_ | :wh | :x: |
| r   | las | ite |     |
|     | t_d | _ch |     |
|     | ist | eck |     |
|     | inc | _ma |     |
|     | t   | rk: |     |
| Exp | is_ | :wh | :wh |
| r   | nul | ite | ite |
|     | l   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | is_ | :wh | :x: |
| r   | uni | ite |     |
|     | que | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | len | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | max | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | mea | :wh | :wh |
| r   | n   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | min | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | n_u | :wh | :wh |
| r   | niq | ite | ite |
|     | ue  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | nul | :wh | :wh |
| r   | l_c | ite | ite |
|     | oun | _ch | _ch |
|     | t   | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | ove | :wh | :x: |
| r   | r   | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | qua | :wh | :x: |
| r   | nti | ite |     |
|     | le  | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | rou | :wh | :x: |
| r   | nd  | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | sam | :wh | :wh |
| r   | ple | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | shi | :wh | :x: |
| r   | ft  | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | sor | :wh | :x: |
| r   | t   | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | std | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | str | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | sum | :wh | :wh |
| r   |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | tai | :wh | :wh |
| r   | l   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Exp | uni | :wh | :x: |
| r   | que | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Exp | get | :wh | :wh |
| rCa | _ca | ite | ite |
| tNa | teg | _ch | _ch |
| mes | ori | eck | eck |
| pac | es  | _ma | _ma |
| e   |     | rk: | rk: |
| Exp | day | :wh | :x: |
| rDa |     | ite |     |
| teT |     | _ch |     |
| ime |     | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | hou | :wh | :x: |
| rDa | r   | ite |     |
| teT |     | _ch |     |
| ime |     | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | mic | :wh | :x: |
| rDa | ros | ite |     |
| teT | eco | _ch |     |
| ime | nd  | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | mil | :wh | :x: |
| rDa | lis | ite |     |
| teT | eco | _ch |     |
| ime | nd  | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | min | :wh | :x: |
| rDa | ute | ite |     |
| teT |     | _ch |     |
| ime |     | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | mon | :wh | :x: |
| rDa | th  | ite |     |
| teT |     | _ch |     |
| ime |     | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | nan | :wh | :x: |
| rDa | ose | ite |     |
| teT | con | _ch |     |
| ime | d   | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | ord | :wh | :x: |
| rDa | ina | ite |     |
| teT | l_d | _ch |     |
| ime | ay  | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | sec | :wh | :x: |
| rDa | ond | ite |     |
| teT |     | _ch |     |
| ime |     | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | to_ | :wh | :wh |
| rDa | str | ite | ite |
| teT | ing | _ch | _ch |
| ime |     | eck | eck |
| Nam |     | _ma | _ma |
| esp |     | rk: | rk: |
| ace |     |     |     |
| Exp | tot | :wh | :x: |
| rDa | al_ | ite |     |
| teT | mic | _ch |     |
| ime | ros | eck |     |
| Nam | eco | _ma |     |
| esp | nds | rk: |     |
| ace |     |     |     |
| Exp | tot | :wh | :x: |
| rDa | al_ | ite |     |
| teT | mil | _ch |     |
| ime | lis | eck |     |
| Nam | eco | _ma |     |
| esp | nds | rk: |     |
| ace |     |     |     |
| Exp | tot | :wh | :x: |
| rDa | al_ | ite |     |
| teT | min | _ch |     |
| ime | ute | eck |     |
| Nam | s   | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | tot | :wh | :x: |
| rDa | al_ | ite |     |
| teT | nan | _ch |     |
| ime | ose | eck |     |
| Nam | con | _ma |     |
| esp | ds  | rk: |     |
| ace |     |     |     |
| Exp | tot | :wh | :x: |
| rDa | al_ | ite |     |
| teT | sec | _ch |     |
| ime | ond | eck |     |
| Nam | s   | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | yea | :wh | :x: |
| rDa | r   | ite |     |
| teT |     | _ch |     |
| ime |     | eck |     |
| Nam |     | _ma |     |
| esp |     | rk: |     |
| ace |     |     |     |
| Exp | con | :wh | :wh |
| rSt | tai | ite | ite |
| rin | ns  | _ch | _ch |
| gNa |     | eck | eck |
| mes |     | _ma | _ma |
| pac |     | rk: | rk: |
| e   |     |     |     |
| Exp | end | :wh | :wh |
| rSt | s_w | ite | ite |
| rin | ith | _ch | _ch |
| gNa |     | eck | eck |
| mes |     | _ma | _ma |
| pac |     | rk: | rk: |
| e   |     |     |     |
| Exp | hea | :x: | :x: |
| rSt | d   |     |     |
| rin |     |     |     |
| gNa |     |     |     |
| mes |     |     |     |
| pac |     |     |     |
| e   |     |     |     |
| Exp | sli | :wh | :wh |
| rSt | ce  | ite | ite |
| rin |     | _ch | _ch |
| gNa |     | eck | eck |
| mes |     | _ma | _ma |
| pac |     | rk: | rk: |
| e   |     |     |     |
| Exp | sta | :wh | :wh |
| rSt | rts | ite | ite |
| rin | _wi | _ch | _ch |
| gNa | th  | eck | eck |
| mes |     | _ma | _ma |
| pac |     | rk: | rk: |
| e   |     |     |     |
| Exp | tai | :x: | :x: |
| rSt | l   |     |     |
| rin |     |     |     |
| gNa |     |     |     |
| mes |     |     |     |
| pac |     |     |     |
| e   |     |     |     |
| Exp | to_ | :wh | :wh |
| rSt | dat | ite | ite |
| rin | eti | _ch | _ch |
| gNa | me  | eck | eck |
| mes |     | _ma | _ma |
| pac |     | rk: | rk: |
| e   |     |     |     |
| Exp | to_ | :wh | :wh |
| rSt | low | ite | ite |
| rin | erc | _ch | _ch |
| gNa | ase | eck | eck |
| mes |     | _ma | _ma |
| pac |     | rk: | rk: |
| e   |     |     |     |
| Exp | to_ | :wh | :wh |
| rSt | upp | ite | ite |
| rin | erc | _ch | _ch |
| gNa | ase | eck | eck |
| mes |     | _ma | _ma |
| pac |     | rk: | rk: |
| e   |     |     |     |
| Laz | clo | :wh | :wh |
| yFr | ne  | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | col | :wh | :wh |
| yFr | lec | ite | ite |
| ame | t   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | col | :wh | :wh |
| yFr | lec | ite | ite |
| ame | t_s | _ch | _ch |
|     | che | eck | eck |
|     | ma  | _ma | _ma |
|     |     | rk: | rk: |
| Laz | col | :wh | :wh |
| yFr | umn | ite | ite |
| ame | s   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | dro | :wh | :wh |
| yFr | p   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | dro | :wh | :wh |
| yFr | p_n | ite | ite |
| ame | ull | _ch | _ch |
|     | s   | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | fil | :wh | :wh |
| yFr | ter | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | gro | :wh | :wh |
| yFr | up_ | ite | ite |
| ame | by  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | hea | :wh | :wh |
| yFr | d   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | joi | :wh | :wh |
| yFr | n   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | laz | :wh | :wh |
| yFr | y   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | pip | :x: | :x: |
| yFr | e   |     |     |
| ame |     |     |     |
| Laz | ren | :wh | :wh |
| yFr | ame | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | sch | :wh | :wh |
| yFr | ema | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | sel | :wh | :wh |
| yFr | ect | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | sor | :wh | :wh |
| yFr | t   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | tai | :wh | :wh |
| yFr | l   | ite | ite |
| ame |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | uni | :wh | :x: |
| yFr | que | ite |     |
| ame |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Laz | wit | :wh | :wh |
| yFr | h_c | ite | ite |
| ame | olu | _ch | _ch |
|     | mns | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Laz | wit | :wh | :wh |
| yFr | h_r | ite | ite |
| ame | ow_ | _ch | _ch |
|     | ind | eck | eck |
|     | ex  | _ma | _ma |
|     |     | rk: | rk: |
| Ser | abs | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | ali | :wh | :wh |
| ies | as  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | all | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | any | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | cas | :wh | :wh |
| ies | t   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | cat | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | cou | :wh | :wh |
| ies | nt  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | cum | :wh | :wh |
| ies | _su | ite | ite |
|     | m   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | dif | :wh | :wh |
| ies | f   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | dro | :wh | :x: |
| ies | p_n | ite |     |
|     | ull | _ch |     |
|     | s   | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | dt  | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | dty | :wh | :wh |
| ies | pe  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | fil | :wh | :wh |
| ies | l_n | ite | ite |
|     | ull | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | fil | :wh | :wh |
| ies | ter | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | hea | :wh | :wh |
| ies | d   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | is_ | :wh | :x: |
| ies | bet | ite |     |
|     | wee | _ch |     |
|     | n   | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | is_ | :wh | :x: |
| ies | dup | ite |     |
|     | lic | _ch |     |
|     | ate | eck |     |
|     | d   | _ma |     |
|     |     | rk: |     |
| Ser | is_ | :wh | :wh |
| ies | emp | ite | ite |
|     | ty  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | is_ | :wh | :x: |
| ies | fir | ite |     |
|     | st_ | _ch |     |
|     | dis | eck |     |
|     | tin | _ma |     |
|     | ct  | rk: |     |
| Ser | is_ | :wh | :wh |
| ies | in  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | is_ | :wh | :x: |
| ies | las | ite |     |
|     | t_d | _ch |     |
|     | ist | eck |     |
|     | inc | _ma |     |
|     | t   | rk: |     |
| Ser | is_ | :wh | :wh |
| ies | nul | ite | ite |
|     | l   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | is_ | :wh | :x: |
| ies | sor | ite |     |
|     | ted | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | is_ | :wh | :x: |
| ies | uni | ite |     |
|     | que | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | ite | :wh | :wh |
| ies | m   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | len | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | max | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | mea | :wh | :wh |
| ies | n   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | min | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | n_u | :wh | :wh |
| ies | niq | ite | ite |
|     | ue  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | nam | :wh | :wh |
| ies | e   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | nul | :wh | :wh |
| ies | l_c | ite | ite |
|     | oun | _ch | _ch |
|     | t   | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | qua | :wh | :x: |
| ies | nti | ite |     |
|     | le  | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | rou | :wh | :x: |
| ies | nd  | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | sam | :wh | :wh |
| ies | ple | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | sha | :wh | :wh |
| ies | pe  | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | shi | :wh | :x: |
| ies | ft  | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | sor | :wh | :x: |
| ies | t   | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | std | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | str | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | sum | :wh | :wh |
| ies |     | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | tai | :wh | :wh |
| ies | l   | ite | ite |
|     |     | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | to_ | :wh | :x: |
| ies | fra | ite |     |
|     | me  | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | to_ | :wh | :wh |
| ies | lis | ite | ite |
|     | t   | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | to_ | :wh | :wh |
| ies | num | ite | ite |
|     | py  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | to_ | :wh | :x: |
| ies | pan | ite |     |
|     | das | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | uni | :wh | :x: |
| ies | que | ite |     |
|     |     | _ch |     |
|     |     | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | val | :wh | :x: |
| ies | ue_ | ite |     |
|     | cou | _ch |     |
|     | nts | eck |     |
|     |     | _ma |     |
|     |     | rk: |     |
| Ser | zip | :wh | :wh |
| ies | _wi | ite | ite |
|     | th  | _ch | _ch |
|     |     | eck | eck |
|     |     | _ma | _ma |
|     |     | rk: | rk: |
| Ser | get | :wh | :wh |
| ies | _ca | ite | ite |
| Cat | teg | _ch | _ch |
| Nam | ori | eck | eck |
| esp | es  | _ma | _ma |
| ace |     | rk: | rk: |
| Ser | day | :wh | :x: |
| ies |     | ite |     |
| Dat |     | _ch |     |
| eTi |     | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | hou | :wh | :x: |
| ies | r   | ite |     |
| Dat |     | _ch |     |
| eTi |     | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | mic | :wh | :x: |
| ies | ros | ite |     |
| Dat | eco | _ch |     |
| eTi | nd  | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | mil | :wh | :x: |
| ies | lis | ite |     |
| Dat | eco | _ch |     |
| eTi | nd  | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | min | :wh | :x: |
| ies | ute | ite |     |
| Dat |     | _ch |     |
| eTi |     | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | mon | :wh | :x: |
| ies | th  | ite |     |
| Dat |     | _ch |     |
| eTi |     | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | nan | :wh | :x: |
| ies | ose | ite |     |
| Dat | con | _ch |     |
| eTi | d   | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | ord | :wh | :x: |
| ies | ina | ite |     |
| Dat | l_d | _ch |     |
| eTi | ay  | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | sec | :wh | :x: |
| ies | ond | ite |     |
| Dat |     | _ch |     |
| eTi |     | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | to_ | :wh | :wh |
| ies | str | ite | ite |
| Dat | ing | _ch | _ch |
| eTi |     | eck | eck |
| meN |     | _ma | _ma |
| ame |     | rk: | rk: |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | tot | :wh | :x: |
| ies | al_ | ite |     |
| Dat | mic | _ch |     |
| eTi | ros | eck |     |
| meN | eco | _ma |     |
| ame | nds | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | tot | :wh | :x: |
| ies | al_ | ite |     |
| Dat | mil | _ch |     |
| eTi | lis | eck |     |
| meN | eco | _ma |     |
| ame | nds | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | tot | :wh | :x: |
| ies | al_ | ite |     |
| Dat | min | _ch |     |
| eTi | ute | eck |     |
| meN | s   | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | tot | :wh | :x: |
| ies | al_ | ite |     |
| Dat | nan | _ch |     |
| eTi | ose | eck |     |
| meN | con | _ma |     |
| ame | ds  | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | tot | :wh | :x: |
| ies | al_ | ite |     |
| Dat | sec | _ch |     |
| eTi | ond | eck |     |
| meN | s   | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | yea | :wh | :x: |
| ies | r   | ite |     |
| Dat |     | _ch |     |
| eTi |     | eck |     |
| meN |     | _ma |     |
| ame |     | rk: |     |
| spa |     |     |     |
| ce  |     |     |     |
| Ser | con | :wh | :wh |
| ies | tai | ite | ite |
| Str | ns  | _ch | _ch |
| ing |     | eck | eck |
| Nam |     | _ma | _ma |
| esp |     | rk: | rk: |
| ace |     |     |     |
| Ser | end | :wh | :wh |
| ies | s_w | ite | ite |
| Str | ith | _ch | _ch |
| ing |     | eck | eck |
| Nam |     | _ma | _ma |
| esp |     | rk: | rk: |
| ace |     |     |     |
| Ser | hea | :x: | :x: |
| ies | d   |     |     |
| Str |     |     |     |
| ing |     |     |     |
| Nam |     |     |     |
| esp |     |     |     |
| ace |     |     |     |
| Ser | sli | :wh | :wh |
| ies | ce  | ite | ite |
| Str |     | _ch | _ch |
| ing |     | eck | eck |
| Nam |     | _ma | _ma |
| esp |     | rk: | rk: |
| ace |     |     |     |
| Ser | sta | :wh | :wh |
| ies | rts | ite | ite |
| Str | _wi | _ch | _ch |
| ing | th  | eck | eck |
| Nam |     | _ma | _ma |
| esp |     | rk: | rk: |
| ace |     |     |     |
| Ser | tai | :x: | :x: |
| ies | l   |     |     |
| Str |     |     |     |
| ing |     |     |     |
| Nam |     |     |     |
| esp |     |     |     |
| ace |     |     |     |
| Ser | to_ | :wh | :wh |
| ies | low | ite | ite |
| Str | erc | _ch | _ch |
| ing | ase | eck | eck |
| Nam |     | _ma | _ma |
| esp |     | rk: | rk: |
| ace |     |     |     |
| Ser | to_ | :wh | :wh |
| ies | upp | ite | ite |
| Str | erc | _ch | _ch |
| ing | ase | eck | eck |
| Nam |     | _ma | _ma |
| esp |     | rk: | rk: |
| ace |     |     |     |