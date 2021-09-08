# Climate Change Uncertainty Spillover in the Macroeconomy
This repository contains codes and jupyter notebooks which compute and demonstrate results of "Climate Change Uncertainty Spillover in the Macroeconomy" by [Micheal Barnett][id1], [William Brock][id2] and [Lars Peter Hansen][id3]. Latest version could be found [here][id4].

Check out online demonstration: [![Read the Docs](https://img.shields.io/readthedocs/pip)](https://ClimateUncertaintySpillover.readthedocs.io/en/latest/)

We also have a jupyter book version: [!(https://img.shields.io/badge/jupyterbook-welcome-brightgreen)](https://lphansen.github.io/ClimateUncertaintySpillover/)

[id1]: https://wpcarey.asu.edu/people/profile/3490536
[id2]: https://economics.missouri.edu/people/brock
[id3]: https://larspeterhansen.org/
[id4]: http://larspeterhansen.org/wp-content/uploads/2021/07/BBHannualLPH-16.pdf

## Table of contents
- [Prerequisite](#prerequisite)
- [Acessing our project](#acessing)
- [Quick user's guide](#quick-guide)

## <a name="prerequisite"></a>Prerequisite
`Python == 3.8.x`, package manager such as `pip`,  and `jupyter notebook` to checkout the notebooks. 

To go to the documentation of this project, please go to: (link avaliable when published).

This project has been tested in an environment with
> `Python == 3.8.11` and  `jupyter notebook == 5.7.10`

## <a name="acessing"></a>Acessing our project


To store the notebook as well as codes in your local machine. You can do this by following steps below:

1.  Open a Windows command prompt, Mac terminal or Linux terminal and change into the folder you would like to store the files.
 	-  You can do this using the command `cd` in the command prompt (on Windows) or terminal (on Mac and Linux).
        - For example, running `cd 'C:\Users\username\python'` (don’t forget '' around the path name to use an absolute path) would lead me to my designated folder.
     
    ```bash
    $ cd [folder path name]
    ```

2.  Clone the github repository for the paper
    - If you don’t have github installed, try installing it from this page: https://git-scm.com/download.
    - You can do this by running below in the command prompt:
    
    ```bash
    $ git clone https://github.com/lphansen/ClimateUncertaintySpillover.git
    ```
    
3.  Change directories into the ‘Wrestling’ folder and install the required packages for the current user or your initiated virtual environment:
    
    ```bash
    $ cd ClimateUncertaintySpillover
    $ pip install -r requirements.txt
    ```
4. Access the notebooks, run the following under the folder `ClimateUncertaintySpillover/`:
    
    ```bash
    $ jupyter notebook
    ```
    and you will see the notebooks listed below.

    Then click <button type = "button" name="runall">Run all</button> from <button type="runtime" name="runtime">runtime</button> to see our results. 

## <a name="quick-guide"></a>Quick user's guide

description of notebooks' contents

The notebooks are separated according to section in the paper.
- Section 1: [Introduction](sec1_Introduction.ipynb)
- Section 2: [Uncertain climate dynamics](sec2_UncertainClimateDynamics.ipynb)
- Section 3: [Uncertain damage](sec3_UncertainDamage.ipynb)
- Section 4: [Illustrative economy I: uncertain damages](sec4_IllustrativeEconIA.ipynb) and [smooth ambiguity](sec4_IllustrativeEconIB.ipynb)
- Section 5: [Illustrative economy II: uncertainty decomposition](sec5_IllustrativeEconII.ipynb)
- Section 6: [Illustrative economy III: carbon abatement technology](sec6_IllustrativeEconIII.ipynb)
- Section 7: [Illustrative economy IV: tail-end damages](sec7_IllustrativeEconIV.ipynb)
- Section 8: [Conclusion](sec8_Conclusion.ipynb)

As well as appendices:
- Appendix A: [Complete model](appendixA.ipynb)
- Appendix B: [Computational methods](appendixB.ipynb)
- Appendix C: [When Y has two states](appendixC.ipynb)
