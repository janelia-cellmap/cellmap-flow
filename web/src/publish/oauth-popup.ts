// oauth.html's entry: the Sign in with Hugging Face popup (oauth.ts says how
// the flow goes). The page shows one line, which this replaces with what the
// user should know.
import { message } from "../shared/message";
import { runSignInPopup } from "./oauth";

const line = document.getElementById("message")!;
const say = (text: string) => {
  line.textContent = text;
};
runSignInPopup(say).catch((e: unknown) => say(`The sign-in could not start: ${message(e)}`));
