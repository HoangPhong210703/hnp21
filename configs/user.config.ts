import type { UserConfig } from "../src/site.config";

const userConfig: UserConfig = {
  title: "hnp's Blog",
  description:
    "A personal blog about programming, technology, and career.",

  url: "https://hnp21.hoangphong210703.workers.dev",
  author: "hnp",

  logo: "/logo.svg",
  avatar: "/avatar.png",

  navigation: [
    { title: "Writing", url: "/posts" },
    { title: "Archive", url: "/archive" },
    { title: "About", url: "/about" },
  ],

  footerLinks: [
    // { title: "RSS", url: "/rss.xml" },
    { title: "Archive", url: "/archive" },
    // { title: "Source", url: "https://github.com/HoangPhong21" },
    
  ],

  social: [
    {
      title: "GitHub",
      url: "https://github.com/HoangPhong21",
      icon: "github",
    },
    // {
    //   title: "X",
    //   url: "https://x.com/HoangPhong21",
    //   icon: "x",
    // },
    // {
    //   title: "LinkedIn",
    //   url: "https://linkedin.com/in/hoang-nguyen-phong",
    //   icon: "linkedin",
    // },
    
  ],

  footerCredits: "Designed for reading.",

  postsPerPage: 8,
  recentPosts: 6,
  relatedPosts: 4,

  showThemeToggle: true,
  showReadingTime: true,

  heroVariant: "studio",

  // annotation: "Writing between filter coffees and terminal windows.",
};

export default userConfig;