# Security

## Reporting a Vulnerability

If you believe you have found a security vulnerability in Narwhals, please report it
privately rather than opening a public issue. We support two channels:

* **GitHub Private Vulnerability Reporting**: use the
  [Report a vulnerability](https://github.com/narwhals-dev/narwhals/security/advisories/new)
  form on our Security tab.
* **Email**: send details to `hello_narwhals@proton.me`.

Please include a description of the issue, reproduction steps, the affected version(s),
and any proof-of-concept code. Do not include data that is itself sensitive.

### Disclosure process and timeline

* We acknowledge receipt of vulnerability reports within 7 days.
* We aim to provide an initial assessment (accepted, more info needed, or declined) within 30 days.
* For confirmed vulnerabilities we target a fix and coordinated disclosure within 90 days
  of the initial report. We will keep the reporter informed of progress and may request
  an embargo extension for complex issues.
* After a fix is released, we publish a GitHub Security Advisory and credit the reporter
  unless they prefer to remain anonymous.

We follow good-faith vulnerability disclosure: researchers acting in line with this
policy will not be pursued under DMCA, CFAA, or equivalent local statutes.

## Security practices

Given that Narwhals can only work if people trust it, we recognise the importance of following
good security practices. Here are some practices we follow:

* We publish to PyPI via trusted publishing and are PEP740-compliant.
* We don't use `pull_request_target` in any CI job.
* The release CI job can only be triggered for tag pushes, and only
  Narwhals members with release permissions (see below) can push tags.
* All members of `narwhals-dev` are required to have two-factor authentication enabled.
* There are no binary or opaque files in the Narwhals repository.
* Release permissions are only given to people who satisfy all of the following:

    * Have met the original author in real life on multiple days.
    * Have made significant contributions to Narwhals.
    * Give off good vibes. This is hard to rigorously define, but it's there so we
        can refuse anyone who, despite satisfying the above two criteria, we don't
        feel like we can trust.
    * There are fewer than 5 active people with release permissions. That is
        to say, even if someone satisfies all of the above, if there are already 5
        people with release permissions, then we will not be adding any more (though
        you may still be added to `narwhals-dev` and get permission to merge pull
        requests which you believe are ready). Note that we already meet that limit.
