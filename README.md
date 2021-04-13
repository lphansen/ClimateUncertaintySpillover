# Climate Change Uncertainty Spillover in the Macroeconomy
This repository contains codes and jupyter notebooks which compute and demonstrate results of "Climate Change Uncertainty Spillover in the Macroeconomy" by [Micheal Barnett][id1], [William Brock][id2] and [Lars Peter Hansen][id3]. Latest version could be found [here][id4].

[id1]: tocomplete
[id2]: https://economics.missouri.edu/people/brock
[id3]: https://larspeterhansen.org/
[id4]: https://larspeterhansen.org/research/papers/

## Table of contents
- [Prerequisite](#prerequisite)
- [Acessing our project](#acessing)
- [Quick user's guide](#quick-guide)

## <a name="prerequisite"></a>Prerequisite
`Python == 3.8.x`, package manager such as `pip`,  and `jupyter notebook` to checkout the notebooks. 

To go to the documentation of this project, please go to: (link avaliable when published).

This project has been tested in an environment with
> `Python == 3.8.7` and  `jupyter notebook == 5.7.10`

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
    $ git clone https://github.com/lphansen/WrestlingClimatie.git
    ```
    
3.  Change directories into the ‘Wrestling’ folder and install the required packages for the current user or your initiated virtual environment:
    
    ```bash
    $ cd WrestlingClimate
    $ pip install -r requirements.txt
    ```
4. Access the notebooks, run the following under the folder `WrestlingClimate/`:
    
    ```bash
    $ jupyter notebook
    ```
    and you will see the notebooks listed below.

    Then click <button type = "button" name="runall">Run all</button> from <button type="runtime" name="runtime">runtime</button> to see our results. 

## <a name="quick-guide"></a>Quick user's guide

description of notebooks' contents

The notebooks are separated according to section in the paper.
- Section 1 - 4: Introduction to example economy
- Section 5: [Uncertainty aversion](sec5_UncertaintyAversion.ipynb)
- Section 6: [Climate components of a planner's decision problem](sec6_DecisionProblem.ipynb)
- Section 7: [Sensitivity](sec7_Sensitivity.ipynb)
- Section 8: [Uncertainty decomposition](sec8_UncertaintyDecomposition.ipynb)
- Section 9: [A richer economic setting](sec9_RicherSetting.ipynb)

As well as appendices:
- A1.
