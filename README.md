# Wrestling with Uncertainty in Climate Economic Models
This repository contains codes and jupyter notebooks which compute and demonstrate results of the example economy section in "Wrestling with Uncertainty in Climate Economic Models" by [William Brock][id1] and [Lars Peter Hansen][id2]. Latest version could be found [here][id4].

[id1]: https://economics.missouri.edu/people/brock
[id2]: https://larspeterhansen.org/
[id4]: https://larspeterhansen.org/research/papers/

## Table of contents
- [Prerequisite](#prerequisite)
- [Acessing our project](#acessing)
- [Quick user's guide](#quick-guide)
- [Snapshots of results](#snapshot)

## <a name="prerequisite"></a>Prerequisite
`Python == 3.8.x`, package manager such as `pip` and `jupyterlab`. 

## <a name="acessing"></a>Acessing our project
There are two options to access our jupyter notebook. The easiest way is to open a copy in Google Colab by clicking the button below:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lphansen/Beliefs/blob/master/Belief_Notebook.ipynb)

Then click <button type = "button" name="runall">Run all</button> from <button type="runtime" name="runtime">runtime</button> to see our results. If you are running the notebook the first time, you will need to click the authorization link under the first code cell and copy paste a pop-up string to the input box under the link.

An alternative way is to store the notebook as well as codes in your local machine. You can do this by following steps below:
1.  Open a Windows command prompt, Mac terminal or Linux terminal and change into the folder you would like to store the files.
    - You can do this using the command ``cd`` in the command prompt (on Windows) or terminal (on Mac and Linux).
    - For example, running `cd 'C:\Users\username\python'` (don’t forget '' around the path name to use an absolute path) would lead me to my designated folder.
    ```
    $ cd [folder path name]
    ```
2.  Clone the github repository for the paper
    - If you don’t have github installed, try installing it from this page: https://git-scm.com/download.
    - You can do this by running below in the command prompt:
    ```
    $ git clone https://github.com/lphansen/WrestlingClimate
    ```
3.	Change directories into the ‘Wrestling’ folder and install the required packages for the current user or your initiated virtual environment:
    ```
    $ cd WrestlingClimate
    $ pip install -r requirements.txt
    ```
## <a name="quick-guide"></a>Quick user's guide
The simulated data are provided in`./data/e_smul` and the pre-generated plots are provided in folder `./notebook/paper_plots/`.
1.  The plots and tables are presented in `./notebook/PaperResults.ipynb`. To re-generate results, open jupyter notebook by running below in command prompt:
    - If you don’t have anaconda3 and jupyter notebook installed, try installing from this page: https://jupyter.org/install.html
    ```
    $ jupyter notebook
    ```
    Open `notebook/PaperResults.ipynb`
    Run notebook cell by cell or click <button type = "button" name = "buttton" class="btn">cell → Run All</button>.
2.  The script for simulation is provided in `./source/simulate.py`. If you want to run the simulation yourself, you may consider entering the following commands:
    ```
    $ cd source
    $ python simulate.py
    ```

    The simulation for each emission path takes around 200s on the test machine.

## <a name="snapshot"></a>Snapshots of results
The emission paths for γ = .018:
<p align="center">
<img src="./notebook/paper_plots/emission_base.png" width="600"/>
</p>

Drift distortion:
<p align="center">
<img src="./notebook/paper_plots/h_hat_base.png" width="600"/>
</p>
