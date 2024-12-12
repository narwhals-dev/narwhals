# Security

Given that Narwhals can only work if people trust it, we recognise the importance of following
good security practices. Here are some practices we follow:

- We publish to PyPI via trusted publishing and are PEP740-compliant.
- We don't use `pull_request_target` in any CI job.
- We sanitise the (potentially unsafe) `github.ref_name` variable when publishing
  releases.
- All members of `narwhals-dev` are required to have two-factor authentication
  enabled.
- Release permissions are only given to people who satisfy all of the following:

    - Have met the original author in real life on multiple days.
    - Have made significant contributions to Narwhals.
    - Give off good vibes. This is hard to rigorously define, but it's there so we
        can refuse anyone who, despite satisfying the above two criteria, we don't
        feel like we can trust.
    - There are fewer than 5 active people with release permissions. That is
        to say, even if someone satisfies all of the above, if there are already 5
        people with release permissions, then we will not be adding any more (though
        you may still be added to `narwhals-dev` and get permission to merge pull
        requests which you believe are ready).
