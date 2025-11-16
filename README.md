# Collective behaviour 2025/2026, group D

For info about the codebase, see [README_DEV.md](./README_DEV.md).

## Group members

- ja8769: [@jan146](https://github.com/jan146/)
- mp4116: [@mpopovic4116](https://github.com/mpopovic4116)
- ah5393: [@anzehocevar](https://github.com/anzehocevar)

## Starting point / article

**Paper:** *Collective phases and long-term dynamics in a fish school model with burst-and-coast swimming*  
**Authors:** Weijia Wang, Ramón Escobedo, Stéphane Sanchez, Zhangang Han, Clément Sire, and Guy Theraulaz  
**Published:** 9 May 2025  
**DOI:** <https://doi.org/10.1098/rsos.240885>

The paper presents a model in which fish alternate between burst and coast phases, adjusting their motion according to local interactions. The authors identify distinct collective phases and transitions depending on model parameters. This will serve as our baseline for replication and extension.

## Planned contributions

We intend to extend the paper by introducing additional parameters to describe the burst phase, and studying their effects on collective behavior.
Currently, the burst phase is an instantaneous event. We intend to add a "duty cycle" (as per real fish) as well as multiple decision points (not biologically grounded, but will help study the effects of asynchronous decision making by approximating synchronous decision making to an arbitrary degree).

## Project plan and milestones

| Report | Deadline | Description |
|--------|----------|-------------|
| ✅ **Report 1: Concept Review and Baseline Setup** | 16 Nov 2025 | Detailed review of existing collective behavior models for fish. Review of literature describing the behavior of real fish to ensure we don't end up modeling something that doesn't happen in nature. |
| ⬜ **Report 2: Methods and Verification Plan** | 7 Dec 2025 | Reproduction of original paper. Proposal for contributions and verification methods following initial experiments. |
| ⬜ **Report 3: Final Report and Presentation** | 11 Jan 2026 | Final polished report. Includes reproduction of original paper, our contribution, verification results, discussion and ideas for future work. |
