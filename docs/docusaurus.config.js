// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Sentiment Analysis Pipeline",
  tagline: "Production-Ready ML/DL Sentiment Classification System",
  favicon: "img/favicon.ico",

  // Set the production url of your site here
  url: "https://Nouman64-cat.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: "/sentiment-production-pipeline/",

  // GitHub pages deployment config.
  organizationName: "Nouman64-cat",
  projectName: "sentiment-production-pipeline",
  trailingSlash: false,
  deploymentBranch: "gh-pages",

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl:
            "https://github.com/Nouman64-cat/sentiment-production-pipeline/tree/main/docs/",
        },
        blog: false, // Disable blog for this project
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: "img/docusaurus-social-card.jpg",
      navbar: {
        title: "Sentiment Pipeline",
        logo: {
          alt: "Sentiment Pipeline Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "tutorialSidebar",
            position: "left",
            label: "Documentation",
          },
          {
            href: "https://github.com/Nouman64-cat/sentiment-production-pipeline",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Documentation",
            items: [
              {
                label: "Getting Started",
                to: "/docs/intro",
              },
              {
                label: "API Reference",
                to: "/docs/api-usage",
              },
            ],
          },
          {
            title: "Technical",
            items: [
              {
                label: "Architecture",
                to: "/docs/architecture",
              },
              {
                label: "Model Choices",
                to: "/docs/model-choices",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "GitHub",
                href: "https://github.com/Nouman64-cat/sentiment-production-pipeline",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Sentiment Analysis Pipeline. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ["python", "bash", "json"],
      },
    }),
};

module.exports = config;
