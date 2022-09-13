# Third-party libraries

This directory contains libraries manually imported from other place.
Code in this directory is maintained almost under the *as-is* basis. We don't guarantee
that code is maintained under the same maintenance policy we applied to other parts in
this repository.
We also don't guarantee that the code is up-to-date.

Before uploading any code to this directory, you have to make sure that the change
satisfies **all** of the following conditions:

- If the code is trivial enough, consider implementing a similar logic to our main
  component.
- Code is distributed under a valid open-source license. The license has to be
  [compatible with the MIT license](https://en.wikipedia.org/wiki/Permissive_software_license)
  or [public domain](https://en.wikipedia.org/wiki/Public_domain).
  If you couldn't judge if the license is compatible with MIT or not, please consult
  with us before making any changes.
- Code is not able to be obtained from the PyPI.

If you decided to upload a third-party to this directory, please follow the steps below:

0. Don't use neither `git submodule` nor `git subtree` without special approvals.
1. Make an individual directory to store the new third-party.
2. Put code and license file onto the directory.
3. If the code contains `README.md`, rename it to `README.orig.md`.
4. Add `README.md`. It has to contain the following information:
   - The place where the original code is distributed.
   - The date when the original code is obtained.
   - If you applied some patch, please add details of it.
5. Request a code review:
   - If you are adding a new third-party code, please create a separate pull request from
     any other changes.
   - If you are updating an existing third-party code, the pull request may contain
     related changes under other parts of this repository.
