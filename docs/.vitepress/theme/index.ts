import DefaultTheme from "vitepress/theme";
import PythonRunner from "./components/PythonRunner.vue";
import "./style.css";

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component("PythonRunner", PythonRunner);
  },
};
