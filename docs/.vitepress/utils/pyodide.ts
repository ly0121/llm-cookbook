let pyodideInstance: any = null;
let loadingPromise: Promise<any> | null = null;

export async function loadPyodide(): Promise<any> {
  if (pyodideInstance) return pyodideInstance;

  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    // @ts-ignore
    const { loadPyodide: load } = await import(
      "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.mjs"
    );
    pyodideInstance = await load({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/",
    });

    // 预装 numpy
    await pyodideInstance.loadPackage("numpy");

    return pyodideInstance;
  })();

  return loadingPromise;
}

export async function runPython(
  code: string
): Promise<{ output: string; error: string }> {
  const pyodide = await loadPyodide();

  // 捕获 stdout 和 stderr
  pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
  `);

  let output = "";
  let error = "";

  try {
    const result = pyodide.runPython(code);
    output = pyodide.runPython("sys.stdout.getvalue()");
    const stderr = pyodide.runPython("sys.stderr.getvalue()");
    if (stderr) error = stderr;

    // 如果代码有返回值且没有 print 输出，显示返回值
    if (!output && result !== undefined && result !== null) {
      output = String(result);
    }
  } catch (e: any) {
    error = e.message || String(e);
  } finally {
    // 重置 stdout/stderr
    pyodide.runPython(`
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
    `);
  }

  return { output: output.trimEnd(), error: error.trimEnd() };
}

export function isPyodideLoaded(): boolean {
  return pyodideInstance !== null;
}
