/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    "intro",
    {
      type: "category",
      label: "Getting Started",
      items: ["installation", "architecture"],
    },
    {
      type: "category",
      label: "Data & Preprocessing",
      items: ["data-curation", "dataset", "preprocessing"],
    },
    {
      type: "category",
      label: "Models",
      items: ["model-choices", "results", "evaluation"],
    },
    {
      type: "category",
      label: "API",
      items: ["api-usage"],
    },
  ],
};

module.exports = sidebars;
