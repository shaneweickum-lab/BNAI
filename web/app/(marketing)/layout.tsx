import Header from "../../components/Header";
import Footer from "../../components/Footer";

// The marketing pages (/ and /about) keep the site chrome; /demo does not
// -- it's a full-viewport app shell (sidebar + chat + showcase drawers),
// not a page with a header/footer, so it lives outside this route group.
export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Header />
      <main style={{ flex: 1 }}>{children}</main>
      <Footer />
    </>
  );
}
