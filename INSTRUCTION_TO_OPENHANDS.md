# Your Role
You are an sophisticated git project manager with plenty experiences in development.

# My Request
Please revise the /workspace/mngs package.
- Please update README.md files, including those in submodules (/workspace/mngs/src/mngs/*)
- Please update docstrings, alinging the style explained below.
- Reorganize package structure if necessary.
- Please write test code under /workspace/mngs/tests, with the same structure of the source code (/workspace/mngs/src/mngs/)
- Please create a sphinx document, which is hosted at https://ywatanabe1989.github.io/mngs/.

# Echo
Please add newline before echo. You might want to use echo-nl instead of echo.

``` bash
function echo-nl() {
    echo -e "\n$@"
}
```


# Setup environment
Activate python environment:

``` bash
source /workspace/env/bin/activate
pip install -e /workspace/mngs_repo
```

# Git
- You can use git. GITHUB_TOKEN is set as an environmental variable. 
- Always ensure your working branch is named openhands; otherwise first create openhands branch and swith to it.
- As long as the branch is openahands, you can use git commands and gh commands.
For mode details, see below
- /home/ywatanabe/.bash.d/all/030-git/*.sh
- /home/ywatanabe/.bash.d/all/030-github/*.sh


# Programming Rules
- Avoid unnecessary comments as they are disruptive. 
	- Return only the updated code without comments.

- Avoid 1-letter variable, as seaching them is challenging
  - For example, rename variable x to xx to balance readability, writability, and searchability.
  
- Well-written code is self-explanatory; variable, function, and class names are crucial.
	- Ultimately, comments can be distracting if the code is properly written.

- Please do not forget docstring with an example usage.
  
- Your output should be code block(s) with a language identifier:
  ``` python
  (DEBUGGED CODE HERE)
  ```

- Use a modular approach for better handling, utilizing functions, classes, and similar structures.

- Use the following docstring styles.
	- For Python, use the NumPy style:
        ``` python
        def func(arg1, arg2):
            """Summary line.

            Extended description of function.

            Example
            ----------
            x, y = 1, 2
            out = func(x, y)
			print(out)

            Parameters
            ----------
            arg1 : int
                Description of arg1
            arg2 : str
                Description of arg2

            Returns
            -------
            bool
                Description of return value

            """
            return True
        ```
    
	- For shell script, please provide example usage at the first line of a function.
        ``` bash
        my-echo() {
          # print the arguments with my signature

          echo "$@"" (Yusuke Watanabe)"
        }
        ```
