#include "tcp_gateway.h"

#include <boost/asio.hpp>

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "../matching_engine/matching_main.h"

namespace ghostbook::gateway {
namespace {

namespace asio = boost::asio;
using boost::system::error_code;
using tcp = asio::ip::tcp;
using udp = asio::ip::udp;

matching::Side to_matching_side(protocol::Side side) {
    return side == protocol::Side::Buy ? matching::Side::Buy : matching::Side::Sell;
}

matching::OrderType to_matching_order_type(protocol::OrderType order_type) {
    return order_type == protocol::OrderType::Market ? matching::OrderType::Market : matching::OrderType::Limit;
}

matching::TimeInForce to_matching_tif(protocol::TimeInForce tif) {
    switch (tif) {
        case protocol::TimeInForce::IOC:
            return matching::TimeInForce::IOC;
        case protocol::TimeInForce::FOK:
            return matching::TimeInForce::FOK;
        case protocol::TimeInForce::PostOnly:
            return matching::TimeInForce::PostOnly;
        case protocol::TimeInForce::Day:
        default:
            return matching::TimeInForce::Day;
    }
}

void set_exec_type_and_status(matching::EventType type, protocol::ExecutionReportBody* report, bool partial_fill) {
    switch (type) {
        case matching::EventType::Ack:
            report->exec_type = static_cast<std::uint8_t>(protocol::ExecType::Ack);
            report->ord_status = static_cast<std::uint8_t>(protocol::OrdStatus::New);
            break;
        case matching::EventType::Fill:
            report->exec_type = static_cast<std::uint8_t>(partial_fill ? protocol::ExecType::PartialFill : protocol::ExecType::Fill);
            report->ord_status = static_cast<std::uint8_t>(partial_fill ? protocol::OrdStatus::PartiallyFilled : protocol::OrdStatus::Filled);
            break;
        case matching::EventType::Cancel:
            report->exec_type = static_cast<std::uint8_t>(protocol::ExecType::Cancel);
            report->ord_status = static_cast<std::uint8_t>(
                report->leaves_qty_u32 == 0 ? protocol::OrdStatus::Canceled : protocol::OrdStatus::PartiallyFilled);
            break;
        case matching::EventType::Reject:
            report->exec_type = static_cast<std::uint8_t>(protocol::ExecType::Reject);
            report->ord_status = static_cast<std::uint8_t>(protocol::OrdStatus::Rejected);
            break;
    }
}

protocol::MessageType message_type_from_exec_type(std::uint8_t exec_type) {
    switch (static_cast<protocol::ExecType>(exec_type)) {
        case protocol::ExecType::Ack:
            return protocol::MessageType::Ack;
        case protocol::ExecType::Fill:
            return protocol::MessageType::Fill;
        case protocol::ExecType::PartialFill:
            return protocol::MessageType::PartialFill;
        case protocol::ExecType::Cancel:
            return protocol::MessageType::Cancel;
        case protocol::ExecType::Reject:
        default:
            return protocol::MessageType::Reject;
    }
}

template <typename T, std::size_t Capacity>
class SpscQueue {
public:
    static_assert(Capacity > 1, "Capacity must be > 1");

    bool push(const T& value) {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto next = (head + 1) % Capacity;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        data_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T* out) {
        const auto tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        *out = data_[tail];
        tail_.store((tail + 1) % Capacity, std::memory_order_release);
        return true;
    }

private:
    std::array<T, Capacity> data_{};
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};

std::vector<std::uint8_t> build_frame(
    protocol::MessageType msg_type,
    const std::uint8_t* body,
    std::uint16_t body_len,
    std::uint64_t seq_no,
    std::uint64_t logical_clock,
    std::uint32_t session_id,
    std::uint8_t flags = 0
) {
    std::vector<std::uint8_t> frame(protocol::header_size + body_len);
    auto header = protocol::make_frame_header(msg_type, body_len, seq_no, logical_clock, session_id, flags);
    std::memcpy(frame.data(), &header, protocol::header_size);
    if (body_len > 0 && body != nullptr) {
        std::memcpy(frame.data() + protocol::header_size, body, body_len);
    }
    return frame;
}

}  // namespace

class TcpGateway::Impl {
public:
    explicit Impl(TcpGateway* owner, logical_clock_t start_clock)
        : owner_(owner),
          io_context_(),
          work_guard_(asio::make_work_guard(io_context_)),
          order_acceptor_(io_context_),
          snapshot_acceptor_(io_context_),
          md_socket_(io_context_),
          outbound_pump_timer_(io_context_),
          running_(false),
          outbound_seq_no_(1),
          market_data_seq_no_(1),
          exchange_order_id_counter_(1000),
          matching_engine_(start_clock) {}

    ~Impl() {
        shutdown();
    }

    bool start(
        std::uint16_t port,
        const std::string& market_data_host,
        std::uint16_t market_data_port,
        std::uint16_t snapshot_port
    ) {
        if (running_.exchange(true, std::memory_order_acq_rel)) {
            return false;
        }

        error_code ec;
        setup_md_socket(market_data_host, market_data_port, &ec);
        if (ec) {
            running_.store(false, std::memory_order_release);
            return false;
        }

        setup_acceptor(order_acceptor_, port, &ec);
        if (ec) {
            running_.store(false, std::memory_order_release);
            return false;
        }

        setup_acceptor(snapshot_acceptor_, snapshot_port, &ec);
        if (ec) {
            running_.store(false, std::memory_order_release);
            return false;
        }

        start_accept_orders();
        start_accept_snapshot();
        schedule_outbound_pump();

        io_thread_ = std::thread([this]() {
            io_context_.run();
        });

        engine_thread_ = std::thread([this]() {
            engine_loop();
        });

        return true;
    }

    void shutdown() {
        if (!running_.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        io_context_.post([this]() {
            error_code ec;
            outbound_pump_timer_.cancel(ec);
            order_acceptor_.close(ec);
            snapshot_acceptor_.close(ec);
            md_socket_.close(ec);
            for (auto& [_, session] : sessions_) {
                session->socket.close(ec);
            }
            sessions_.clear();
            sessions_by_id_.clear();
        });

        work_guard_.reset();
        io_context_.stop();

        if (io_thread_.joinable()) {
            io_thread_.join();
        }

        if (engine_thread_.joinable()) {
            engine_thread_.join();
        }
    }

    void process_frame_inline(
        session_id_t session_id,
        const protocol::FrameHeader& header,
        const std::vector<std::uint8_t>& body
    ) {
        auto type = static_cast<protocol::MessageType>(header.msg_type);
        switch (type) {
            case protocol::MessageType::NewOrder:
                inline_new_order(session_id, body);
                break;
            case protocol::MessageType::CancelOrder:
                inline_cancel_order(session_id, body);
                break;
            case protocol::MessageType::ModifyOrder:
                inline_modify_order(session_id, body);
                break;
            case protocol::MessageType::HeartBeat:
                break;
            case protocol::MessageType::Logoff:
                owner_->session_mgr_.close_session(session_id);
                break;
            default:
                owner_->stats_.protocol_errors++;
                break;
        }

        owner_->logical_clock_ = matching_engine_.logical_clock();
    }

private:
    struct Session : public std::enable_shared_from_this<Session> {
        explicit Session(asio::io_context& io) : socket(io), read_chunk(4096, 0) {}

        tcp::socket socket;
        FrameBuffer buffer;
        std::vector<std::uint8_t> read_chunk;
        bool authenticated{false};
        session_id_t session_id{0};
        std::uint64_t seq_no{1};
        std::deque<std::vector<std::uint8_t>> write_queue;
        bool write_in_progress{false};
    };

    enum class CommandType : std::uint8_t {
        NewOrder,
        CancelOrder,
        ModifyOrder,
        HeartBeat,
        Logoff
    };

    struct EngineCommand {
        CommandType type{CommandType::HeartBeat};
        session_id_t session_id{0};
        protocol::NewOrderBody new_order{};
        protocol::CancelOrderBody cancel_order{};
        protocol::ModifyOrderBody modify_order{};
    };

    struct EngineReport {
        session_id_t session_id{0};
        protocol::ExecutionReportBody report{};
        logical_clock_t logical_clock{0};
    };

    struct InstrumentMarketDataState {
        std::map<std::int64_t, std::uint64_t, std::greater<std::int64_t>> bid_qty_by_price;
        std::map<std::int64_t, std::uint64_t, std::less<std::int64_t>> ask_qty_by_price;
    };

    static constexpr std::size_t kQueueCapacity = 1 << 15;

    TcpGateway* owner_;

    asio::io_context io_context_;
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;

    tcp::acceptor order_acceptor_;
    tcp::acceptor snapshot_acceptor_;
    udp::socket md_socket_;
    udp::endpoint md_endpoint_;
    bool md_is_multicast_{false};

    asio::steady_timer outbound_pump_timer_;

    std::unordered_map<std::uintptr_t, std::shared_ptr<Session>> sessions_;
    std::unordered_map<session_id_t, std::weak_ptr<Session>> sessions_by_id_;

    std::atomic<bool> running_;
    std::thread io_thread_;
    std::thread engine_thread_;

    SpscQueue<EngineCommand, kQueueCapacity> command_queue_;
    SpscQueue<EngineReport, kQueueCapacity> report_queue_;

    std::atomic<std::uint64_t> outbound_seq_no_;
    std::atomic<std::uint64_t> market_data_seq_no_;
    std::uint64_t exchange_order_id_counter_;

    matching::MatchingEngine matching_engine_;
    std::map<order_id_t, session_id_t> order_owner_session_;
    std::map<order_id_t, order_id_t> client_to_exchange_id_;

    std::map<order_id_t, protocol::AddOrderMdBody> market_data_orders_;
    std::map<instrument_id_t, InstrumentMarketDataState> market_data_books_;

    void setup_acceptor(tcp::acceptor& acceptor, std::uint16_t port, error_code* ec) {
        acceptor.open(tcp::v4(), *ec);
        if (*ec) {
            return;
        }

        acceptor.set_option(asio::socket_base::reuse_address(true), *ec);
        if (*ec) {
            return;
        }

        acceptor.bind(tcp::endpoint(tcp::v4(), port), *ec);
        if (*ec) {
            return;
        }

        acceptor.listen(asio::socket_base::max_listen_connections, *ec);
    }

    void setup_md_socket(const std::string& host, std::uint16_t port, error_code* ec) {
        md_socket_.open(udp::v4(), *ec);
        if (*ec) {
            return;
        }

        md_endpoint_ = udp::endpoint(asio::ip::make_address(host, *ec), port);
        if (*ec) {
            return;
        }

        md_is_multicast_ = md_endpoint_.address().is_multicast();
        if (md_is_multicast_) {
            md_socket_.set_option(asio::ip::multicast::hops(1), *ec);
        }
    }

    void start_accept_orders() {
        auto session = std::make_shared<Session>(io_context_);
        order_acceptor_.async_accept(session->socket, [this, session](const error_code& ec) {
            if (!ec && running_.load(std::memory_order_acquire)) {
                const auto key = reinterpret_cast<std::uintptr_t>(session.get());
                sessions_[key] = session;
                start_read(session);
            }
            if (running_.load(std::memory_order_acquire)) {
                start_accept_orders();
            }
        });
    }

    void start_accept_snapshot() {
        auto sock = std::make_shared<tcp::socket>(io_context_);
        snapshot_acceptor_.async_accept(*sock, [this, sock](const error_code& ec) {
            if (!ec && running_.load(std::memory_order_acquire)) {
                send_snapshot(*sock);
            }
            if (running_.load(std::memory_order_acquire)) {
                start_accept_snapshot();
            }
        });
    }

    void start_read(const std::shared_ptr<Session>& session) {
        auto self = session;
        session->socket.async_read_some(
            asio::buffer(session->read_chunk),
            [this, self](const error_code& ec, std::size_t bytes_transferred) {
                if (ec || bytes_transferred == 0) {
                    close_session(self);
                    return;
                }

                try {
                    self->buffer.append(self->read_chunk.data(), bytes_transferred);
                } catch (...) {
                    close_session(self);
                    return;
                }

                while (true) {
                    auto frame = self->buffer.try_read_frame();
                    if (!frame.has_value()) {
                        break;
                    }

                    const auto& [header, body] = *frame;
                    handle_session_frame(self, header, body);
                    owner_->stats_.total_messages_received++;
                }

                if (running_.load(std::memory_order_acquire)) {
                    start_read(self);
                }
            });
    }

    void handle_session_frame(
        const std::shared_ptr<Session>& session,
        const protocol::FrameHeader& header,
        const std::vector<std::uint8_t>& body
    ) {
        const auto type = static_cast<protocol::MessageType>(header.msg_type);

        if (type == protocol::MessageType::Logon) {
            if (session->authenticated) {
                return;
            }
            handle_logon(session, header, body);
            return;
        }

        if (!session->authenticated) {
            send_unauthenticated_reject(session);
            return;
        }

        if (type == protocol::MessageType::Logoff) {
            enqueue_command({.type = CommandType::Logoff, .session_id = session->session_id});
            owner_->session_mgr_.close_session(session->session_id);
            sessions_by_id_.erase(session->session_id);
            session->authenticated = false;
            return;
        }

        if (type == protocol::MessageType::HeartBeat) {
            enqueue_command({.type = CommandType::HeartBeat, .session_id = session->session_id});
            return;
        }

        EngineCommand command{};
        command.session_id = session->session_id;

        switch (type) {
            case protocol::MessageType::NewOrder:
                if (body.size() < sizeof(protocol::NewOrderBody)) {
                    owner_->stats_.protocol_errors++;
                    return;
                }
                command.type = CommandType::NewOrder;
                std::memcpy(&command.new_order, body.data(), sizeof(protocol::NewOrderBody));
                owner_->stats_.total_orders_submitted++;
                enqueue_command(command);
                break;
            case protocol::MessageType::CancelOrder:
                if (body.size() < sizeof(protocol::CancelOrderBody)) {
                    owner_->stats_.protocol_errors++;
                    return;
                }
                command.type = CommandType::CancelOrder;
                std::memcpy(&command.cancel_order, body.data(), sizeof(protocol::CancelOrderBody));
                owner_->stats_.total_cancels_submitted++;
                enqueue_command(command);
                break;
            case protocol::MessageType::ModifyOrder:
                if (body.size() < sizeof(protocol::ModifyOrderBody)) {
                    owner_->stats_.protocol_errors++;
                    return;
                }
                command.type = CommandType::ModifyOrder;
                std::memcpy(&command.modify_order, body.data(), sizeof(protocol::ModifyOrderBody));
                owner_->stats_.total_modifies_submitted++;
                enqueue_command(command);
                break;
            default:
                owner_->stats_.protocol_errors++;
                break;
        }
    }

    void handle_logon(
        const std::shared_ptr<Session>& session,
        const protocol::FrameHeader& header,
        const std::vector<std::uint8_t>& body
    ) {
        if (body.size() < sizeof(protocol::LogonBody)) {
            owner_->stats_.protocol_errors++;
            return;
        }

        protocol::LogonBody logon{};
        std::memcpy(&logon, body.data(), sizeof(logon));

        const session_id_t sid = owner_->session_mgr_.create_session(
            logon.comp_id,
            logon.app_instance,
            logon.client_ip_v4,
            logon.heartbeat_interval_ms);

        session->authenticated = true;
        session->session_id = sid;
        sessions_by_id_[sid] = session;

        protocol::ExecutionReportBody ack{};
        ack.exec_type = static_cast<std::uint8_t>(protocol::ExecType::Ack);
        ack.ord_status = static_cast<std::uint8_t>(protocol::OrdStatus::New);

        auto frame = build_frame(
            protocol::MessageType::Ack,
            reinterpret_cast<const std::uint8_t*>(&ack),
            static_cast<std::uint16_t>(sizeof(ack)),
            header.seq_no + 1,
            owner_->logical_clock_,
            sid);
        queue_write(session, std::move(frame));
    }

    void send_unauthenticated_reject(const std::shared_ptr<Session>& session) {
        protocol::ExecutionReportBody reject{};
        reject.exec_type = static_cast<std::uint8_t>(protocol::ExecType::Reject);
        reject.ord_status = static_cast<std::uint8_t>(protocol::OrdStatus::Rejected);
        reject.code_u16 = 401;

        auto frame = build_frame(
            protocol::MessageType::Reject,
            reinterpret_cast<const std::uint8_t*>(&reject),
            static_cast<std::uint16_t>(sizeof(reject)),
            outbound_seq_no_.fetch_add(1, std::memory_order_relaxed),
            owner_->logical_clock_,
            0);
        queue_write(session, std::move(frame));
    }

    void queue_write(const std::shared_ptr<Session>& session, std::vector<std::uint8_t>&& frame) {
        session->write_queue.push_back(std::move(frame));
        if (!session->write_in_progress) {
            do_write(session);
        }
    }

    void do_write(const std::shared_ptr<Session>& session) {
        if (session->write_queue.empty()) {
            session->write_in_progress = false;
            return;
        }

        session->write_in_progress = true;
        auto self = session;
        asio::async_write(
            session->socket,
            asio::buffer(session->write_queue.front()),
            [this, self](const error_code& ec, std::size_t) {
                if (ec) {
                    close_session(self);
                    return;
                }
                self->write_queue.pop_front();
                do_write(self);
            });
    }

    void close_session(const std::shared_ptr<Session>& session) {
        error_code ec;
        session->socket.close(ec);

        if (session->session_id != 0) {
            sessions_by_id_.erase(session->session_id);
            owner_->session_mgr_.close_session(session->session_id);
        }

        sessions_.erase(reinterpret_cast<std::uintptr_t>(session.get()));
    }

    void enqueue_command(const EngineCommand& command) {
        while (running_.load(std::memory_order_acquire) && !command_queue_.push(command)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    void enqueue_report(const EngineReport& report) {
        while (running_.load(std::memory_order_acquire) && !report_queue_.push(report)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    void engine_loop() {
        while (running_.load(std::memory_order_acquire)) {
            EngineCommand command{};
            if (!command_queue_.pop(&command)) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            process_engine_command(command);
            flush_engine_events();
        }
    }

    void process_engine_command(const EngineCommand& command) {
        switch (command.type) {
            case CommandType::NewOrder: {
                matching::NewOrderCommand c{
                    .client_order_id = command.new_order.client_order_id,
                    .instrument_id = command.new_order.instrument_id,
                    .side = to_matching_side(static_cast<protocol::Side>(command.new_order.side)),
                    .order_type = to_matching_order_type(static_cast<protocol::OrderType>(command.new_order.order_type)),
                    .tif = to_matching_tif(static_cast<protocol::TimeInForce>(command.new_order.time_in_force)),
                    .price_tick = command.new_order.price_i64,
                    .quantity = command.new_order.qty_u32,
                };

                order_owner_session_[c.client_order_id] = command.session_id;
                client_to_exchange_id_[c.client_order_id] = exchange_order_id_counter_++;
                matching_engine_.submit(c);
                break;
            }
            case CommandType::CancelOrder: {
                matching::CancelOrderCommand c{
                    .client_order_id = command.cancel_order.client_order_id,
                    .target_order_id = command.cancel_order.target_order_id,
                    .cancel_quantity = command.cancel_order.cancel_qty_u32,
                };

                order_owner_session_[c.client_order_id] = command.session_id;
                client_to_exchange_id_[c.client_order_id] = exchange_order_id_counter_++;
                matching_engine_.cancel(c);
                break;
            }
            case CommandType::ModifyOrder: {
                matching::ModifyOrderCommand c{
                    .original_order_id = command.modify_order.orig_client_order_id,
                    .new_order_id = command.modify_order.new_client_order_id,
                    .new_price_tick = command.modify_order.new_price_i64,
                    .new_quantity = command.modify_order.new_qty_u32,
                };

                order_owner_session_[c.new_order_id] = command.session_id;
                client_to_exchange_id_[c.new_order_id] = exchange_order_id_counter_++;
                matching_engine_.modify(c);
                break;
            }
            case CommandType::HeartBeat:
            case CommandType::Logoff:
                break;
        }
    }

    void flush_engine_events() {
        const auto events = matching_engine_.drain_events();
        for (const auto& event : events) {
            protocol::ExecutionReportBody report{};
            report.client_order_id = event.order_id;
            report.exchange_order_id = client_to_exchange_id_.contains(event.order_id)
                ? client_to_exchange_id_[event.order_id]
                : event.order_id;
            report.instrument_id = event.instrument_id;
            report.side = static_cast<std::uint8_t>(
                event.side == matching::Side::Buy ? protocol::Side::Buy : protocol::Side::Sell);
            report.last_qty_u32 = event.quantity;
            report.last_price_i64 = event.price_tick;
            report.leaves_qty_u32 = event.remaining_quantity;
            report.cum_qty_u32 = event.cumulative_quantity;
            report.code_u16 = (event.type == matching::EventType::Reject) ? 1 : 0;
            report.liquidity_flag = static_cast<std::uint8_t>(protocol::LiquidityFlag::Taker);

            if (event.type == matching::EventType::Reject) {
                owner_->stats_.total_rejections++;
            }

            const bool partial_fill = event.type == matching::EventType::Fill && event.remaining_quantity > 0;
            set_exec_type_and_status(event.type, &report, partial_fill);

            session_id_t sid = 0;
            const auto it = order_owner_session_.find(event.order_id);
            if (it != order_owner_session_.end()) {
                sid = it->second;
            }

            enqueue_report(EngineReport{
                .session_id = sid,
                .report = report,
                .logical_clock = event.logical_clock,
            });

            if ((event.type == matching::EventType::Fill || event.type == matching::EventType::Cancel) && event.remaining_quantity == 0) {
                order_owner_session_.erase(event.order_id);
            }
        }

        owner_->logical_clock_ = matching_engine_.logical_clock();
    }

    void schedule_outbound_pump() {
        outbound_pump_timer_.expires_after(std::chrono::milliseconds(1));
        outbound_pump_timer_.async_wait([this](const error_code& ec) {
            if (ec || !running_.load(std::memory_order_acquire)) {
                return;
            }

            EngineReport report{};
            while (report_queue_.pop(&report)) {
                handle_outbound_report(report);
            }

            schedule_outbound_pump();
        });
    }

    void handle_outbound_report(const EngineReport& outbound) {
        owner_->logical_clock_ = outbound.logical_clock;

        if (outbound.session_id != 0) {
            auto session_it = sessions_by_id_.find(outbound.session_id);
            if (session_it != sessions_by_id_.end()) {
                if (auto session = session_it->second.lock()) {
                    const auto msg_type = message_type_from_exec_type(outbound.report.exec_type);
                    auto frame = build_frame(
                        msg_type,
                        reinterpret_cast<const std::uint8_t*>(&outbound.report),
                        static_cast<std::uint16_t>(sizeof(outbound.report)),
                        outbound_seq_no_.fetch_add(1, std::memory_order_relaxed),
                        owner_->logical_clock_,
                        outbound.session_id);
                    queue_write(session, std::move(frame));
                }
            }
        }

        if (static_cast<protocol::ExecType>(outbound.report.exec_type) == protocol::ExecType::Ack) {
            owner_->emit_order_ack(
                outbound.session_id,
                outbound.report.client_order_id,
                outbound.report.exchange_order_id,
                outbound.logical_clock);
        }

        owner_->emit_execution_report(outbound.session_id, outbound.report);
        publish_market_data(outbound.report);
    }

    void publish_market_data(const protocol::ExecutionReportBody& report) {
        const auto exec_type = static_cast<protocol::ExecType>(report.exec_type);

        if (exec_type == protocol::ExecType::Ack && report.leaves_qty_u32 > 0 && report.instrument_id != 0) {
            protocol::AddOrderMdBody add{};
            add.md_order_id = report.exchange_order_id;
            add.instrument_id = report.instrument_id;
            add.side = report.side;
            add.price_i64 = report.last_price_i64;
            add.qty_u32 = report.leaves_qty_u32;
            publish_add_order(add, false);
        }

        if (exec_type == protocol::ExecType::Fill || exec_type == protocol::ExecType::PartialFill) {
            protocol::TradeBody trade{};
            trade.md_order_id = report.exchange_order_id;
            trade.instrument_id = report.instrument_id;
            trade.side = report.side;
            trade.price_i64 = report.last_price_i64;
            trade.qty_u32 = report.last_qty_u32;
            send_md_frame(
                protocol::MessageType::Trade,
                reinterpret_cast<const std::uint8_t*>(&trade),
                static_cast<std::uint16_t>(sizeof(trade)),
                false);

            if (report.leaves_qty_u32 == 0) {
                publish_delete_order(report.exchange_order_id, false);
            } else {
                publish_modify_order(report.exchange_order_id, report.leaves_qty_u32, false);
            }
        }

        if (exec_type == protocol::ExecType::Cancel) {
            if (report.leaves_qty_u32 == 0) {
                publish_delete_order(report.exchange_order_id, false);
            } else {
                publish_modify_order(report.exchange_order_id, report.leaves_qty_u32, false);
            }
        }

        if (report.instrument_id != 0) {
            publish_bbo(report.instrument_id, false);
        }
    }

    void send_md_frame(
        protocol::MessageType msg_type,
        const std::uint8_t* body,
        std::uint16_t body_len,
        bool is_snapshot
    ) {
        if (!md_socket_.is_open()) {
            return;
        }

        const auto seq = market_data_seq_no_.fetch_add(1, std::memory_order_relaxed);
        auto frame = std::make_shared<std::vector<std::uint8_t>>(
            build_frame(msg_type, body, body_len, seq, owner_->logical_clock_, 0, is_snapshot ? 0x04 : 0x00));

        md_socket_.async_send_to(
            asio::buffer(*frame),
            md_endpoint_,
            [frame](const error_code&, std::size_t) {
                (void)frame;
            });
    }

    void publish_add_order(const protocol::AddOrderMdBody& add, bool is_snapshot) {
        market_data_orders_[add.md_order_id] = add;
        auto& book = market_data_books_[add.instrument_id];
        if (static_cast<protocol::Side>(add.side) == protocol::Side::Buy) {
            book.bid_qty_by_price[add.price_i64] += add.qty_u32;
        } else {
            book.ask_qty_by_price[add.price_i64] += add.qty_u32;
        }

        send_md_frame(
            protocol::MessageType::AddOrder,
            reinterpret_cast<const std::uint8_t*>(&add),
            static_cast<std::uint16_t>(sizeof(add)),
            is_snapshot);
    }

    void publish_delete_order(order_id_t order_id, bool is_snapshot) {
        const auto it = market_data_orders_.find(order_id);
        if (it == market_data_orders_.end()) {
            return;
        }

        const auto& prev = it->second;
        auto& book = market_data_books_[prev.instrument_id];
        if (static_cast<protocol::Side>(prev.side) == protocol::Side::Buy) {
            auto l = book.bid_qty_by_price.find(prev.price_i64);
            if (l != book.bid_qty_by_price.end()) {
                l->second = (l->second > prev.qty_u32) ? (l->second - prev.qty_u32) : 0;
                if (l->second == 0) {
                    book.bid_qty_by_price.erase(l);
                }
            }
        } else {
            auto l = book.ask_qty_by_price.find(prev.price_i64);
            if (l != book.ask_qty_by_price.end()) {
                l->second = (l->second > prev.qty_u32) ? (l->second - prev.qty_u32) : 0;
                if (l->second == 0) {
                    book.ask_qty_by_price.erase(l);
                }
            }
        }

        protocol::DeleteOrderMdBody del{};
        del.md_order_id = order_id;
        del.instrument_id = prev.instrument_id;
        del.side = prev.side;
        del.remaining_qty_u32 = prev.qty_u32;

        market_data_orders_.erase(it);

        send_md_frame(
            protocol::MessageType::DeleteOrder,
            reinterpret_cast<const std::uint8_t*>(&del),
            static_cast<std::uint16_t>(sizeof(del)),
            is_snapshot);
    }

    void publish_modify_order(order_id_t order_id, std::uint32_t new_qty, bool is_snapshot) {
        const auto it = market_data_orders_.find(order_id);
        if (it == market_data_orders_.end()) {
            return;
        }

        auto& prev = it->second;
        if (new_qty == 0) {
            publish_delete_order(order_id, is_snapshot);
            return;
        }

        auto& book = market_data_books_[prev.instrument_id];
        std::uint64_t* agg = nullptr;
        if (static_cast<protocol::Side>(prev.side) == protocol::Side::Buy) {
            agg = &book.bid_qty_by_price[prev.price_i64];
        } else {
            agg = &book.ask_qty_by_price[prev.price_i64];
        }

        if (new_qty > prev.qty_u32) {
            *agg += (new_qty - prev.qty_u32);
        } else {
            const auto delta = prev.qty_u32 - new_qty;
            *agg = (*agg > delta) ? (*agg - delta) : 0;
        }

        prev.qty_u32 = new_qty;

        protocol::ModifyOrderMdBody mod{};
        mod.md_order_id = order_id;
        mod.instrument_id = prev.instrument_id;
        mod.side = prev.side;
        mod.new_price_i64 = prev.price_i64;
        mod.new_qty_u32 = new_qty;

        send_md_frame(
            protocol::MessageType::ModifyOrderMd,
            reinterpret_cast<const std::uint8_t*>(&mod),
            static_cast<std::uint16_t>(sizeof(mod)),
            is_snapshot);
    }

    void publish_bbo(instrument_id_t instrument_id, bool is_snapshot) {
        auto it = market_data_books_.find(instrument_id);
        if (it == market_data_books_.end()) {
            return;
        }

        protocol::BboBody bbo{};
        bbo.instrument_id = instrument_id;

        const auto& state = it->second;
        if (!state.bid_qty_by_price.empty()) {
            bbo.bid_price_i64 = state.bid_qty_by_price.begin()->first;
            bbo.bid_qty_u32 = static_cast<std::uint32_t>(state.bid_qty_by_price.begin()->second);
        }
        if (!state.ask_qty_by_price.empty()) {
            bbo.ask_price_i64 = state.ask_qty_by_price.begin()->first;
            bbo.ask_qty_u32 = static_cast<std::uint32_t>(state.ask_qty_by_price.begin()->second);
        }

        send_md_frame(
            protocol::MessageType::BBO,
            reinterpret_cast<const std::uint8_t*>(&bbo),
            static_cast<std::uint16_t>(sizeof(bbo)),
            is_snapshot);
    }

    void send_snapshot(tcp::socket& socket) {
        std::vector<std::uint8_t> payload;
        std::uint64_t seq = market_data_seq_no_.load(std::memory_order_relaxed);

        auto append_frame = [&payload](std::vector<std::uint8_t>&& frame) {
            payload.insert(payload.end(), frame.begin(), frame.end());
        };

        for (const auto& [_, order] : market_data_orders_) {
            append_frame(build_frame(
                protocol::MessageType::AddOrder,
                reinterpret_cast<const std::uint8_t*>(&order),
                static_cast<std::uint16_t>(sizeof(order)),
                seq++,
                owner_->logical_clock_,
                0,
                0x04));
        }

        for (const auto& [instrument, _] : market_data_books_) {
            protocol::BboBody bbo{};
            bbo.instrument_id = instrument;
            const auto& state = market_data_books_[instrument];
            if (!state.bid_qty_by_price.empty()) {
                bbo.bid_price_i64 = state.bid_qty_by_price.begin()->first;
                bbo.bid_qty_u32 = static_cast<std::uint32_t>(state.bid_qty_by_price.begin()->second);
            }
            if (!state.ask_qty_by_price.empty()) {
                bbo.ask_price_i64 = state.ask_qty_by_price.begin()->first;
                bbo.ask_qty_u32 = static_cast<std::uint32_t>(state.ask_qty_by_price.begin()->second);
            }

            append_frame(build_frame(
                protocol::MessageType::BBO,
                reinterpret_cast<const std::uint8_t*>(&bbo),
                static_cast<std::uint16_t>(sizeof(bbo)),
                seq++,
                owner_->logical_clock_,
                0,
                0x04));
        }

        protocol::SequenceResetBody watermark{};
        watermark.new_seq_no = market_data_seq_no_.load(std::memory_order_relaxed);
        append_frame(build_frame(
            protocol::MessageType::SequenceReset,
            reinterpret_cast<const std::uint8_t*>(&watermark),
            static_cast<std::uint16_t>(sizeof(watermark)),
            seq,
            owner_->logical_clock_,
            0,
            0x04));

        auto payload_ptr = std::make_shared<std::vector<std::uint8_t>>(std::move(payload));
        asio::async_write(
            socket,
            asio::buffer(*payload_ptr),
            [&socket, payload_ptr](const error_code&, std::size_t) {
                error_code ec;
                socket.shutdown(tcp::socket::shutdown_both, ec);
                socket.close(ec);
            });
    }

    void inline_new_order(session_id_t sid, const std::vector<std::uint8_t>& body) {
        if (body.size() < sizeof(protocol::NewOrderBody)) {
            owner_->stats_.protocol_errors++;
            return;
        }

        protocol::NewOrderBody msg{};
        std::memcpy(&msg, body.data(), sizeof(msg));

        matching::NewOrderCommand c{
            .client_order_id = msg.client_order_id,
            .instrument_id = msg.instrument_id,
            .side = to_matching_side(static_cast<protocol::Side>(msg.side)),
            .order_type = to_matching_order_type(static_cast<protocol::OrderType>(msg.order_type)),
            .tif = to_matching_tif(static_cast<protocol::TimeInForce>(msg.time_in_force)),
            .price_tick = msg.price_i64,
            .quantity = msg.qty_u32,
        };

        order_owner_session_[c.client_order_id] = sid;
        client_to_exchange_id_[c.client_order_id] = exchange_order_id_counter_++;
        matching_engine_.submit(c);
        owner_->stats_.total_orders_submitted++;

        inline_flush_events();
    }

    void inline_cancel_order(session_id_t sid, const std::vector<std::uint8_t>& body) {
        if (body.size() < sizeof(protocol::CancelOrderBody)) {
            owner_->stats_.protocol_errors++;
            return;
        }

        protocol::CancelOrderBody msg{};
        std::memcpy(&msg, body.data(), sizeof(msg));

        matching::CancelOrderCommand c{
            .client_order_id = msg.client_order_id,
            .target_order_id = msg.target_order_id,
            .cancel_quantity = msg.cancel_qty_u32,
        };

        order_owner_session_[c.client_order_id] = sid;
        client_to_exchange_id_[c.client_order_id] = exchange_order_id_counter_++;
        matching_engine_.cancel(c);
        owner_->stats_.total_cancels_submitted++;

        inline_flush_events();
    }

    void inline_modify_order(session_id_t sid, const std::vector<std::uint8_t>& body) {
        if (body.size() < sizeof(protocol::ModifyOrderBody)) {
            owner_->stats_.protocol_errors++;
            return;
        }

        protocol::ModifyOrderBody msg{};
        std::memcpy(&msg, body.data(), sizeof(msg));

        matching::ModifyOrderCommand c{
            .original_order_id = msg.orig_client_order_id,
            .new_order_id = msg.new_client_order_id,
            .new_price_tick = msg.new_price_i64,
            .new_quantity = msg.new_qty_u32,
        };

        order_owner_session_[c.new_order_id] = sid;
        client_to_exchange_id_[c.new_order_id] = exchange_order_id_counter_++;
        matching_engine_.modify(c);
        owner_->stats_.total_modifies_submitted++;

        inline_flush_events();
    }

    void inline_flush_events() {
        const auto events = matching_engine_.drain_events();
        for (const auto& event : events) {
            protocol::ExecutionReportBody report{};
            report.client_order_id = event.order_id;
            report.exchange_order_id = client_to_exchange_id_.contains(event.order_id)
                ? client_to_exchange_id_[event.order_id]
                : event.order_id;
            report.instrument_id = event.instrument_id;
            report.side = static_cast<std::uint8_t>(
                event.side == matching::Side::Buy ? protocol::Side::Buy : protocol::Side::Sell);
            report.last_qty_u32 = event.quantity;
            report.last_price_i64 = event.price_tick;
            report.leaves_qty_u32 = event.remaining_quantity;
            report.cum_qty_u32 = event.cumulative_quantity;
            report.code_u16 = (event.type == matching::EventType::Reject) ? 1 : 0;
            report.liquidity_flag = static_cast<std::uint8_t>(protocol::LiquidityFlag::Taker);

            if (event.type == matching::EventType::Reject) {
                owner_->stats_.total_rejections++;
            }

            const bool partial_fill = event.type == matching::EventType::Fill && event.remaining_quantity > 0;
            set_exec_type_and_status(event.type, &report, partial_fill);

            session_id_t sid = 0;
            const auto it = order_owner_session_.find(event.order_id);
            if (it != order_owner_session_.end()) {
                sid = it->second;
            }

            if (event.type == matching::EventType::Ack) {
                owner_->emit_order_ack(sid, event.order_id, report.exchange_order_id, event.logical_clock);
            }
            owner_->emit_execution_report(sid, report);
        }
    }
};

TcpGateway::TcpGateway(logical_clock_t start_clock)
    : Gateway(start_clock), impl_(std::make_unique<Impl>(this, start_clock)) {}

TcpGateway::~TcpGateway() = default;

bool TcpGateway::start(
    std::uint16_t port,
    const std::string& market_data_host,
    std::uint16_t market_data_port,
    std::uint16_t snapshot_port
) {
    return impl_->start(port, market_data_host, market_data_port, snapshot_port);
}

void TcpGateway::shutdown() {
    impl_->shutdown();
}

void TcpGateway::process_frame(
    session_id_t session_id,
    const protocol::FrameHeader& header,
    const std::vector<std::uint8_t>& body
) {
    impl_->process_frame_inline(session_id, header, body);
}

session_id_t TcpGateway::create_test_session(
    std::uint16_t comp_id,
    std::uint64_t app_instance,
    std::uint32_t client_ip,
    std::uint16_t heartbeat_interval_ms
) {
    return session_mgr_.create_session(comp_id, app_instance, client_ip, heartbeat_interval_ms);
}

}  // namespace ghostbook::gateway
